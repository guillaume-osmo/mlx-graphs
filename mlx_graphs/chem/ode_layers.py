"""
Single-layer + ODE multistep with skip connection — replaces any stacked layers.
Pattern: h_new = h + dt * f(h) per step; identity skip keeps stability and gradient flow.

Parameter count (fundamental): We count ONLY THE SINGLE LAYER PARAMETERS, not n_steps times.
The same f (same weights) is reused at every ODE step — like weight-sharing in an RNN.
So total params = params(f) + params(proj_in) + params(proj_out) etc.; n_steps does NOT
multiply the parameter count. This is drastically different from stacking n_steps
separate layers (which would be n_steps * params(f)).
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn


def ode_skip_step(h: mx.array, f: Callable[[mx.array], mx.array], dt: float) -> mx.array:
    """One ODE step with skip: h_new = h + dt * f(h)."""
    return h + dt * f(h)


def ode_skip_multistep(
    h: mx.array,
    f: Callable[[mx.array], mx.array],
    dt: float,
    n_steps: int,
) -> mx.array:
    """Multistep ODE with skip: h_{k+1} = h_k + dt * f(h_k), k = 0..n_steps-1."""
    for _ in range(n_steps):
        h = ode_skip_step(h, f, dt)
    return h


class ODESkipLayer(nn.Module):
    """
    Single layer + ODE multistep with skip connection.
    Replaces stacked layers: one f (e.g. Linear(dim, dim)) + residual ODE steps.
    Step: h = h + dt * act(f(h)).

    Parameter count: only the single layer f (and activation has no params). The same
    f is applied n_steps times; we do NOT add parameters per step.
    """

    def __init__(
        self,
        dim: int,
        n_steps: int = 4,
        dt: float = 0.25,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self.dt = dt
        self.f = nn.Linear(dim, dim)
        self.activation = getattr(nn, activation) if hasattr(nn, activation) else nn.leaky_relu

    def _velocity(self, h: mx.array) -> mx.array:
        return self.activation(self.f(h))

    def __call__(self, h: mx.array) -> mx.array:
        return ode_skip_multistep(h, self._velocity, self.dt, self.n_steps)


class ODE_MLP(nn.Module):
    """
    Single-layer + ODE skip, multistep (replaces stacked MLP).
    in -> Linear(in, dim) -> [ODE skip multistep] -> Linear(dim, out).
    Use hidden_dims=(128, 64, 32) for MLP 128->64->32->output_dim after ODESkipLayer(128) with n_steps=3.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        n_steps: int = 4,
        dt: float = 0.25,
        dropout: float = 0.0,
        hidden_dims: tuple[int, ...] | None = None,
    ):
        super().__init__()
        # If hidden_dims given (e.g. (128, 64, 32)), use ODESkipLayer(128) with n_steps=3 then MLP 128->64->32->out
        if hidden_dims is not None and len(hidden_dims) >= 1:
            first = hidden_dims[0]
            self.proj_in = nn.Linear(input_dim, first)
            self.ode = ODESkipLayer(first, n_steps=3, dt=dt)
            self.mlp_layers = []
            prev = first
            for i, d in enumerate(hidden_dims[1:]):
                setattr(self, f"mlp_{i}", nn.Linear(prev, d))
                self.mlp_layers.append((prev, d))
                prev = d
            self.proj_out = nn.Linear(prev, output_dim)
            self._hidden_dims = hidden_dims
        else:
            self.proj_in = nn.Linear(input_dim, hidden_dim)
            self.ode = ODESkipLayer(hidden_dim, n_steps=n_steps, dt=dt)
            self.mlp_layers = []
            self.proj_out = nn.Linear(hidden_dim, output_dim)
            self._hidden_dims = None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def __call__(self, x: mx.array, training: bool = True) -> mx.array:
        h = self.proj_in(x)
        if self.dropout is not None and training:
            h = self.dropout(h)
        h = self.ode(h)
        if self._hidden_dims is not None and len(self._hidden_dims) > 1:
            for i in range(len(self._hidden_dims) - 1):
                h = getattr(self, f"mlp_{i}")(h)
                h = nn.leaky_relu(h)
        return self.proj_out(h)


class ODE_GRU(nn.Module):
    """
    Single GRU layer + ODE skip multistep on hidden state (replaces stacked RNN).
    in -> GRU -> h -> [ODE skip multistep(h)] -> out.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_steps: int = 4,
        dt: float = 0.25,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, bias=True)
        self.ode = ODESkipLayer(hidden_dim, n_steps=n_steps, dt=dt)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, seq_len, input_dim)
        seq_out = self.gru(x)
        h_last = seq_out[:, -1, :]
        return self.ode(h_last)


class ODE_CNN(nn.Module):
    """
    Single Conv1d + ODE skip multistep on features (replaces stacked CNN).
    in -> Conv1d -> [ODE skip multistep per position] -> pool.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n_steps: int = 4,
        dt: float = 0.25,
        pool: str = "mean",
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.ode = ODESkipLayer(out_channels, n_steps=n_steps, dt=dt)
        self.pool = pool

    def __call__(self, x: mx.array) -> mx.array:
        # MLX Conv1d expects (batch, seq_len, channels).
        h = self.conv(x)  # (batch, seq_len_out, out_ch)
        batch, seq, ch = h.shape
        h = mx.reshape(h, (batch * seq, ch))
        h = self.ode(h)
        h = mx.reshape(h, (batch, seq, ch))
        if self.pool == "max":
            return mx.max(h, axis=1)
        return mx.mean(h, axis=1)


def num_parameters(module: nn.Module) -> int:
    """
    Count trainable parameters (single count; ODE multistep reuses same params).
    Use this for fair comparison: ODE_MLP with n_steps=4 has the same param count
    as with n_steps=1 — only the single layer is counted.
    """
    from mlx.utils import tree_flatten
    return sum(p.size for _, p in tree_flatten(module.parameters()))
