"""
Single ODE block: compress dynamics into one learnable layer (stability, speed).
Ported from Aromma. Use anywhere we can replace stacked layers with one layer + ODE.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _softplus(x: mx.array) -> mx.array:
    x_clipped = mx.clip(x, -20.0, 20.0)
    return mx.log(1.0 + mx.exp(x_clipped))


class ODEBlock(nn.Module):
    """
    ODE block: d²C/dt² = -K1*(dC/dt) - K2*C. Params from features; Euler or RK4.
    Single layer + integration for stability; use instead of stacking many layers.
    """

    def __init__(
        self,
        input_dim: int,
        n_steps: int = 20,
        dt: float = 0.1,
        solver: Literal["euler", "rk4"] = "rk4",
    ):
        super().__init__()
        self.n_steps = n_steps
        self.dt = dt
        self.solver = solver
        self.param_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 4),
        )

    def _ode_rhs(
        self,
        C: mx.array,
        dC: mx.array,
        K1: mx.array,
        K2: mx.array,
    ) -> tuple[mx.array, mx.array]:
        d2C = -K1 * dC - K2 * C
        return dC, d2C

    def _euler_step(
        self,
        C: mx.array,
        dC: mx.array,
        K1: mx.array,
        K2: mx.array,
    ) -> tuple[mx.array, mx.array]:
        dC_dt, d2C_dt = self._ode_rhs(C, dC, K1, K2)
        return C + dC_dt * self.dt, dC + d2C_dt * self.dt

    def _rk4_step(
        self,
        C: mx.array,
        dC: mx.array,
        K1: mx.array,
        K2: mx.array,
    ) -> tuple[mx.array, mx.array]:
        dt = self.dt
        dC1, d2C1 = self._ode_rhs(C, dC, K1, K2)
        dC2, d2C2 = self._ode_rhs(C + dC1 * dt / 2, dC + d2C1 * dt / 2, K1, K2)
        dC3, d2C3 = self._ode_rhs(C + dC2 * dt / 2, dC + d2C2 * dt / 2, K1, K2)
        dC4, d2C4 = self._ode_rhs(C + dC3 * dt, dC + d2C3 * dt, K1, K2)
        C_new = C + (dC1 + 2 * dC2 + 2 * dC3 + dC4) * dt / 6
        dC_new = dC + (d2C1 + 2 * d2C2 + 2 * d2C3 + d2C4) * dt / 6
        return C_new, dC_new

    def __call__(self, h: mx.array) -> mx.array:
        """h: (batch, input_dim). Returns (batch, 3): C_final, dC_final, C_integral."""
        params = self.param_net(h)
        K1 = mx.clip(_softplus(params[:, 0:1]) + 0.01, 0.01, 10.0)
        K2 = mx.clip(_softplus(params[:, 1:2]) + 0.01, 0.01, 10.0)
        C0 = mx.sigmoid(params[:, 2:3])
        dC0 = mx.tanh(params[:, 3:4]) * 0.5
        C, dC = C0, dC0
        C_sum = C
        step_fn = self._rk4_step if self.solver == "rk4" else self._euler_step
        for _ in range(self.n_steps):
            C, dC = step_fn(C, dC, K1, K2)
            C = mx.clip(C, -10.0, 10.0)
            dC = mx.clip(dC, -10.0, 10.0)
            C_sum = C_sum + C
        C_integral = C_sum * self.dt
        out = mx.concatenate([C, dC, C_integral], axis=-1)
        return mx.clip(out, -100.0, 100.0)
