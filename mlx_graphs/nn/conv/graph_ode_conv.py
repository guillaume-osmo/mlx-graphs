# Copyright © 2023-2024 Apple Inc.
# Graph ODE: continuous dynamics dh/dt = f(h, graph) for physics-inspired modeling
# (e.g. solubility / diffusion-like phenomena). Integrate with Euler, Heun, Midpoint, or RK4.
# ODE can be applied on top of any GNN: velocity = GNN_step(h) -> h_new = h + dt*velocity.

from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.nn.mol_attention_readout import MolAttentionReadout
from mlx_graphs.utils import scatter

# Integrator choices: "euler" (default), "heun", "midpoint", "rk4"
INTEGRATORS = ("euler", "heun", "midpoint", "rk4")
# Update mode: "ode" = h + dt*(GNN(h)-h); "additive" = h + GNN(h) (simple skip)
UPDATE_MODES = ("ode", "additive")
# Learnable dt: Δt = DT_MIN + (DT_MAX - DT_MIN) * σ(α), so Δt ∈ [DT_MIN, DT_MAX]
DT_MIN, DT_MAX = 0.1, 0.5


class MLPODEBlock(nn.Module):
    """MLP-ODE: integrate in the *first* MLP dim (same layer / same weights per step), only last projection to 1 is separate.
    Like GNN-ODE but for the head: h0 = project(x), h_{t+1} = h_t + dt*(velocity(h_t) - h_t), out = final(h_T).
    Use same dim for all ODE steps (first_mlp_dim, e.g. 64); final Linear(hidden_dim, 1) is the only separate part."""

    def __init__(
        self,
        mlp_in: int,
        hidden_dim: int,
        num_steps: int,
        dt: float = 0.25,
        integrator: str = "euler",
    ):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.project = Linear(mlp_in, hidden_dim, bias=True)
        self.velocity = Linear(hidden_dim, hidden_dim, bias=True)
        self.final = Linear(hidden_dim, 1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.project(x)

        def step_fn(h_in: mx.array) -> mx.array:
            return nn.leaky_relu(self.velocity(h_in))

        h = _ode_integrate(h, self.dt, self.num_steps, step_fn, self.integrator)
        return self.final(h)


def _dt_from_alpha(alpha: mx.array) -> mx.array:
    """Bounded learnable step size: Δt = 0.1 + 0.4 * σ(α) ∈ [0.1, 0.5]."""
    return DT_MIN + (DT_MAX - DT_MIN) * mx.sigmoid(alpha)


def _additive_integrate(
    h: mx.array,
    num_steps: int,
    step_fn: Callable[[mx.array], mx.array],
) -> mx.array:
    """Simple skip: h_{t+1} = h_t + GNN(h_t). step_fn(h) returns GNN(h)."""
    for _ in range(num_steps):
        h = h + step_fn(h)
    return h


def _ode_integrate(
    h: mx.array,
    dt: float,
    num_steps: int,
    step_fn: Callable[[mx.array], mx.array],
    integrator: str,
    norm_fn: Optional[Callable[[mx.array], mx.array]] = None,
) -> mx.array:
    """Integrate dh/dt = step_fn(h) - h (residual form) over num_steps. step_fn(h) returns one GNN step output.
    If norm_fn is set (e.g. LayerNorm), applied after each step (DeeperGATGNN-style)."""
    if integrator not in INTEGRATORS:
        raise ValueError(f"integrator must be one of {INTEGRATORS}, got {integrator!r}")
    for _ in range(num_steps):
        if integrator == "euler":
            h = h + dt * (step_fn(h) - h)
        elif integrator == "heun":
            k1 = step_fn(h) - h
            h_mid = h + dt * k1
            k2 = step_fn(h_mid) - h_mid
            h = h + (dt / 2.0) * (k1 + k2)
        elif integrator == "midpoint":
            k1 = step_fn(h) - h
            h_mid = h + (dt / 2.0) * k1
            k2 = step_fn(h_mid) - h_mid
            h = h + dt * k2
        else:  # rk4
            k1 = step_fn(h) - h
            h2 = h + (dt / 2.0) * k1
            k2 = step_fn(h2) - h2
            h3 = h + (dt / 2.0) * k2
            k3 = step_fn(h3) - h3
            h4 = h + dt * k3
            k4 = step_fn(h4) - h4
            h = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if norm_fn is not None:
            h = norm_fn(h)
    return h


def _ode_integrate_with_dts(
    h: mx.array,
    dt_array: mx.array,
    step_fn: Callable[[mx.array], mx.array],
    integrator: str,
    norm_fn: Optional[Callable[[mx.array], mx.array]] = None,
) -> mx.array:
    """Like _ode_integrate but with one learnable dt per step. dt_array shape (num_steps,).
    If norm_fn is set, applied after each step (DeeperGATGNN-style)."""
    if integrator not in INTEGRATORS:
        raise ValueError(f"integrator must be one of {INTEGRATORS}, got {integrator!r}")
    num_steps = dt_array.shape[0]
    for t in range(num_steps):
        dt = dt_array[t]
        if integrator == "euler":
            h = h + dt * (step_fn(h) - h)
        elif integrator == "heun":
            k1 = step_fn(h) - h
            h_mid = h + dt * k1
            k2 = step_fn(h_mid) - h_mid
            h = h + (dt / 2.0) * (k1 + k2)
        elif integrator == "midpoint":
            k1 = step_fn(h) - h
            h_mid = h + (dt / 2.0) * k1
            k2 = step_fn(h_mid) - h_mid
            h = h + dt * k2
        else:  # rk4
            k1 = step_fn(h) - h
            h2 = h + (dt / 2.0) * k1
            k2 = step_fn(h2) - h2
            h3 = h + (dt / 2.0) * k2
            k3 = step_fn(h3) - h3
            h4 = h + dt * k3
            k4 = step_fn(h4) - h4
            h = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if norm_fn is not None:
            h = norm_fn(h)
    return h


def _ode_step_one(
    h: mx.array,
    dt: float,
    step_fn: Callable[[mx.array], mx.array],
    integrator: str,
) -> mx.array:
    """One ODE step: h_new = h + dt*(...) using step_fn. Used when step_fn differs per step (e.g. ODEDiffGNN)."""
    if integrator == "euler":
        return h + dt * (step_fn(h) - h)
    elif integrator == "heun":
        k1 = step_fn(h) - h
        h_mid = h + dt * k1
        k2 = step_fn(h_mid) - h_mid
        return h + (dt / 2.0) * (k1 + k2)
    elif integrator == "midpoint":
        k1 = step_fn(h) - h
        h_mid = h + (dt / 2.0) * k1
        k2 = step_fn(h_mid) - h_mid
        return h + dt * k2
    else:  # rk4
        k1 = step_fn(h) - h
        h2 = h + (dt / 2.0) * k1
        k2 = step_fn(h2) - h2
        h3 = h + (dt / 2.0) * k2
        k3 = step_fn(h3) - h3
        h4 = h + dt * k3
        k4 = step_fn(h4) - h4
        return h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class _ODEMessagePassing(MessagePassing):
    """Velocity field: message = MLP(concat(x_src, x_dst, e)); aggregate sum."""

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(
            Linear(node_dim * 2 + edge_dim, out_dim),
            nn.ReLU(),
            Linear(out_dim, out_dim),
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
    ) -> mx.array:
        return self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_features": edge_features},
        )

    def message(
        self,
        src_features: mx.array,
        dst_features: mx.array,
        **kwargs,
    ) -> mx.array:
        edge_features = kwargs.get("edge_features")
        if edge_features is None:
            raise ValueError("_ODEMessagePassing requires edge_features")
        inp = mx.concatenate([src_features, dst_features, edge_features], axis=-1)
        return self.mlp(inp)

    def update_nodes(self, aggregated: mx.array, **kwargs) -> mx.array:
        return aggregated


# Norm between steps: "layer" = LayerNorm after each step (DeeperGATGNN-style); None = no norm
NORM_BETWEEN_STEPS = ("layer",)


def _sample_nodes_for_pooling(
    node_repr: mx.array,
    batch_indices: mx.array,
    training: bool,
    subset_k: int = 0,
    subset_ratio: float = 1.0,
    exclude_ends: bool = False,
    rescale_for_sum: bool = False,
) -> tuple[mx.array, mx.array]:
    """Train-time node subset sampling before graph pooling.
    - subset_k > 0: sample exactly k nodes per graph (or all if graph has < k)
    - else subset_ratio < 1.0: sample ceil(ratio * n_nodes_graph), min 1
    - exclude_ends: drop first/last node positions per graph from candidate set when possible.
    """
    if (not training) or (subset_k <= 0 and subset_ratio >= 1.0):
        return node_repr, batch_indices

    b_np = np.array(batch_indices)
    if b_np.size == 0:
        return node_repr, batch_indices
    num_graphs = int(b_np.max()) + 1
    selected_idx = []
    selected_scale = []
    for g in range(num_graphs):
        idx = np.where(b_np == g)[0]
        if idx.size == 0:
            continue
        cand = idx
        if exclude_ends and idx.size > 2:
            cand = idx[1:-1]
            if cand.size == 0:
                cand = idx
        if subset_k > 0:
            n_pick = min(subset_k, cand.size)
        else:
            n_pick = max(1, int(np.ceil(cand.size * subset_ratio)))
            n_pick = min(n_pick, cand.size)
        if n_pick < cand.size:
            pick = np.random.choice(cand, size=n_pick, replace=False)
            pick = np.sort(pick)
        else:
            pick = cand
        selected_idx.append(pick)
        if rescale_for_sum:
            selected_scale.append(np.full((pick.size,), float(idx.size) / float(pick.size), dtype=np.float32))
    if not selected_idx:
        return node_repr, batch_indices
    selected = np.concatenate(selected_idx, axis=0)
    selected_mx = mx.array(selected.astype(np.int32))
    node_sel = node_repr[selected_mx]
    if rescale_for_sum:
        scale = np.concatenate(selected_scale, axis=0)
        scale_mx = mx.array(scale)[:, None]
        node_sel = node_sel * scale_mx
    return node_sel, batch_indices[selected_mx]


class GraphODEBlock(nn.Module):
    """ODE: h + dt*(f(h)-h). Or additive skip: h + GNN(h). update_mode: ode (default) | additive.
    learnable_dt: if True, one Δt per step with Δt = 0.1 + 0.8*σ(α) ∈ [0.1, 0.9].
    norm_between_steps: 'layer' = LayerNorm after each step (DeeperGATGNN-style); None = no norm."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_steps: int = 4,
        dt: float = 0.25,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.update_mode = update_mode
        self.learnable_dt = learnable_dt
        self.norm_between_steps = norm_between_steps
        self.velocity_net = _ODEMessagePassing(hidden_dim, edge_dim, hidden_dim)
        self.node_proj = Linear(node_dim, hidden_dim)
        if norm_between_steps == "layer":
            self.step_norm = nn.LayerNorm(hidden_dim)
        else:
            self.step_norm = None
        if learnable_dt:
            self.dt_alpha = mx.zeros((num_steps,))

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features)
        def step_fn_ode(h_in: mx.array) -> mx.array:
            return h_in + self.velocity_net(edge_index, h_in, edge_features)
        def step_fn_add(h_in: mx.array) -> mx.array:
            return self.velocity_net(edge_index, h_in, edge_features)
        norm_fn = (lambda h: self.step_norm(h)) if self.step_norm is not None else None
        if self.update_mode == "additive":
            h = _additive_integrate(h, self.num_steps, step_fn_add)
        elif self.learnable_dt:
            dts = _dt_from_alpha(self.dt_alpha)
            h = _ode_integrate_with_dts(h, dts, step_fn_ode, self.integrator, norm_fn=norm_fn)
        else:
            h = _ode_integrate(h, self.dt, self.num_steps, step_fn_ode, self.integrator, norm_fn=norm_fn)
        return nn.relu(h)


class GraphODERegressor(nn.Module):
    """Graph ODE regressor: ODE block (continuous dynamics) + global pool + MLP.
    integrator: euler (default), heun, midpoint, rk4. learnable_dt: one Δt per step in [0.1, 0.9].
    norm_between_steps: 'layer' for LayerNorm after each step (DeeperGATGNN-style).
    mol_attention_steps: if > 0, use AttFP-style mol GRU/attention readout instead of sum pool.
    mlp_ode_steps: if > 0, use MLP-ODE (integrate in first MLP dim; only last proj to 1 separate)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: list[int] = (128, 64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        mlp_ode_steps: int = 0,
        mlp_ode_dt: float = 0.25,
        mlp_ode_integrator: str = "euler",
        pool_subset_k: int = 0,
        pool_subset_ratio: float = 1.0,
        pool_exclude_ends: bool = False,
    ):
        super().__init__()
        self.mol_attention_steps = mol_attention_steps
        self.mlp_ode_steps = mlp_ode_steps
        self.ode_block = GraphODEBlock(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_steps=ode_steps,
            dt=ode_dt,
            integrator=integrator,
            update_mode=update_mode,
            learnable_dt=learnable_dt,
            norm_between_steps=norm_between_steps,
        )
        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        if mlp_ode_steps > 0:
            first_dim = mlp_units[0] if mlp_units else 128
            self.mlp_ode = MLPODEBlock(mlp_in, first_dim, mlp_ode_steps, mlp_ode_dt, mlp_ode_integrator)
            self.mlp_layers = None
        else:
            self.mlp_ode = None
            units = [mlp_in] + list(mlp_units) + [1]
            layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
            self.mlp_layers = layers
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.pool_subset_k = pool_subset_k
        self.pool_subset_ratio = pool_subset_ratio
        self.pool_exclude_ends = pool_exclude_ends

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        node_repr = self.ode_block(edge_index, node_features, edge_features, training=training)
        node_pool, batch_pool = _sample_nodes_for_pooling(
            node_repr,
            batch_indices,
            training=training,
            subset_k=self.pool_subset_k,
            subset_ratio=self.pool_subset_ratio,
            exclude_ends=self.pool_exclude_ends,
            rescale_for_sum=True,
        )
        if self.mol_readout is not None:
            graph_repr = self.mol_readout(node_pool, batch_pool, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(node_pool, batch_pool, out_size=num_graphs, aggr="add")
        if self.rdkit_dim > 0 and graph_features is not None:
            h = mx.concatenate([graph_repr, graph_features], axis=-1)
        else:
            h = graph_repr
        if training:
            h = self.dropout(h)
        if self.mlp_ode is not None:
            return self.mlp_ode(h)
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < len(self.mlp_layers) - 1:
                h = nn.leaky_relu(h)
        return h


# --- ODE with different GNN per step (Euler/Heun/... on layer_0, layer_1, ...) ---
class GraphODEBlockDiffGNN(nn.Module):
    """ODE integration with a different GNN at each step: h_{t+1} = h_t + dt*(GNN_t(h_t)-h_t).
    Radial-style depth (different weights per step) with ODE residual form.
    learnable_dt: if True, one Δt per step with Δt = 0.1 + 0.8*σ(α) ∈ [0.1, 0.9].
    norm_between_steps: 'layer' = LayerNorm after each step (DeeperGATGNN-style)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_steps: int = 4,
        dt: float = 0.25,
        integrator: str = "euler",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.learnable_dt = learnable_dt
        self.norm_between_steps = norm_between_steps
        self.node_proj = Linear(node_dim, hidden_dim)
        if norm_between_steps == "layer":
            self.step_norm = nn.LayerNorm(hidden_dim)
        else:
            self.step_norm = None
        if learnable_dt:
            self.dt_alpha = mx.zeros((num_steps,))
        for t in range(num_steps):
            setattr(
                self,
                f"velocity_net_{t}",
                _ODEMessagePassing(hidden_dim, edge_dim, hidden_dim),
            )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features)
        for t in range(self.num_steps):
            net_t = getattr(self, f"velocity_net_{t}")
            dt_t = _dt_from_alpha(self.dt_alpha[t]) if self.learnable_dt else self.dt

            def step_fn(h_in: mx.array, n=net_t) -> mx.array:
                return h_in + n(edge_index, h_in, edge_features)

            h = _ode_step_one(h, dt_t, step_fn, self.integrator)
            if self.step_norm is not None:
                h = self.step_norm(h)
        return nn.relu(h)


class GraphODEDiffGNNRegressor(nn.Module):
    """Regressor: ODE with different GNN per step (GraphODEBlockDiffGNN) + pool + MLP."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: list[int] = (64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        pool_subset_k: int = 0,
        pool_subset_ratio: float = 1.0,
        pool_exclude_ends: bool = False,
    ):
        super().__init__()
        self.ode_block = GraphODEBlockDiffGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_steps=ode_steps,
            dt=ode_dt,
            integrator=integrator,
            learnable_dt=learnable_dt,
            norm_between_steps=norm_between_steps,
        )
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        units = [mlp_in] + list(mlp_units) + [1]
        layers = []
        for i in range(len(units) - 1):
            layers.append(Linear(units[i], units[i + 1], bias=True))
        self.mlp_layers = layers
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.pool_subset_k = pool_subset_k
        self.pool_subset_ratio = pool_subset_ratio
        self.pool_exclude_ends = pool_exclude_ends

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        node_repr = self.ode_block(edge_index, node_features, edge_features, training=training)
        node_pool, batch_pool = _sample_nodes_for_pooling(
            node_repr, batch_indices, training,
            self.pool_subset_k, self.pool_subset_ratio, self.pool_exclude_ends, True,
        )
        num_graphs = int(mx.max(batch_indices).item()) + 1
        graph_repr = scatter(node_pool, batch_pool, out_size=num_graphs, aggr="add")
        if self.rdkit_dim > 0 and graph_features is not None:
            h = mx.concatenate([graph_repr, graph_features], axis=-1)
        else:
            h = graph_repr
        if training:
            h = self.dropout(h)
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < len(self.mlp_layers) - 1:
                h = nn.leaky_relu(h)
        return h


# --- ODE with different GAT per step ---
class GraphODEBlockDiffGAT(nn.Module):
    """ODE with a different GAT layer at each step: h_{t+1} = h_t + dt*(GAT_t(h_t)-h_t).
    learnable_dt: if True, one Δt per step with Δt = 0.1 + 0.8*σ(α) ∈ [0.1, 0.9].
    norm_between_steps: 'layer' = LayerNorm after each step (DeeperGATGNN-style)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        num_steps: int = 4,
        dt: float = 0.25,
        dropout: float = 0.1,
        integrator: str = "euler",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
    ):
        super().__init__()
        assert hidden_dim % heads == 0
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.learnable_dt = learnable_dt
        self.norm_between_steps = norm_between_steps
        self.node_proj = Linear(node_dim, hidden_dim)
        if norm_between_steps == "layer":
            self.step_norm = nn.LayerNorm(hidden_dim)
        else:
            self.step_norm = None
        if learnable_dt:
            self.dt_alpha = mx.zeros((num_steps,))
        GATConv = _get_gat_conv()
        for t in range(num_steps):
            setattr(
                self,
                f"gat_{t}",
                GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_features_dim=edge_dim,
                ),
            )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features)
        for t in range(self.num_steps):
            gat_t = getattr(self, f"gat_{t}")
            dt_t = _dt_from_alpha(self.dt_alpha[t]) if self.learnable_dt else self.dt

            def step_fn(h_in: mx.array, g=gat_t) -> mx.array:
                return nn.leaky_relu(g(edge_index, h_in, edge_features))

            h = _ode_step_one(h, dt_t, step_fn, self.integrator)
            if self.step_norm is not None:
                h = self.step_norm(h)
        return nn.relu(h)


class GraphODEDiffGATRegressor(nn.Module):
    """Regressor: ODE with different GAT per step + pool + MLP."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: list[int] = (64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        pool_subset_k: int = 0,
        pool_subset_ratio: float = 1.0,
        pool_exclude_ends: bool = False,
    ):
        super().__init__()
        self.ode_block = GraphODEBlockDiffGAT(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            num_steps=ode_steps,
            dt=ode_dt,
            dropout=dropout,
            integrator=integrator,
            learnable_dt=learnable_dt,
            norm_between_steps=norm_between_steps,
        )
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        units = [mlp_in] + list(mlp_units) + [1]
        self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.pool_subset_k = pool_subset_k
        self.pool_subset_ratio = pool_subset_ratio
        self.pool_exclude_ends = pool_exclude_ends

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        node_repr = self.ode_block(edge_index, node_features, edge_features, training=training)
        node_pool, batch_pool = _sample_nodes_for_pooling(
            node_repr, batch_indices, training,
            self.pool_subset_k, self.pool_subset_ratio, self.pool_exclude_ends, True,
        )
        num_graphs = int(mx.max(batch_indices).item()) + 1
        graph_repr = scatter(node_pool, batch_pool, out_size=num_graphs, aggr="add")
        h = mx.concatenate([graph_repr, graph_features], axis=-1) if (
            self.rdkit_dim > 0 and graph_features is not None
        ) else graph_repr
        if training:
            h = self.dropout(h)
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < len(self.mlp_layers) - 1:
                h = nn.leaky_relu(h)
        return h


# --- ODE with different GINE per step ---
class GraphODEBlockDiffGINE(nn.Module):
    """ODE with a different GINE layer at each step: h_{t+1} = h_t + dt*(GINE_t(h_t)-h_t).
    learnable_dt: if True, one Δt per step with Δt = 0.1 + 0.8*σ(α) ∈ [0.1, 0.9].
    norm_between_steps: 'layer' = LayerNorm after each step (DeeperGATGNN-style)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_steps: int = 4,
        dt: float = 0.25,
        integrator: str = "euler",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.learnable_dt = learnable_dt
        self.norm_between_steps = norm_between_steps
        self.node_proj = Linear(node_dim, hidden_dim)
        if norm_between_steps == "layer":
            self.step_norm = nn.LayerNorm(hidden_dim)
        else:
            self.step_norm = None
        if learnable_dt:
            self.dt_alpha = mx.zeros((num_steps,))
        GINConv = _get_gin_conv()
        for t in range(num_steps):
            gine_mlp = nn.Sequential(
                Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                Linear(hidden_dim, hidden_dim),
            )
            setattr(
                self,
                f"gine_{t}",
                GINConv(
                    gine_mlp,
                    edge_features_dim=edge_dim,
                    node_features_dim=hidden_dim,
                ),
            )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features)
        for t in range(self.num_steps):
            gine_t = getattr(self, f"gine_{t}")
            dt_t = _dt_from_alpha(self.dt_alpha[t]) if self.learnable_dt else self.dt

            def step_fn(h_in: mx.array, g=gine_t) -> mx.array:
                return nn.relu(g(edge_index, h_in, edge_features))

            h = _ode_step_one(h, dt_t, step_fn, self.integrator)
            if self.step_norm is not None:
                h = self.step_norm(h)
        return nn.relu(h)


class GraphODEDiffGINERegressor(nn.Module):
    """Regressor: ODE with different GINE per step + pool + MLP."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: list[int] = (64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        pool_subset_k: int = 0,
        pool_subset_ratio: float = 1.0,
        pool_exclude_ends: bool = False,
    ):
        super().__init__()
        self.ode_block = GraphODEBlockDiffGINE(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_steps=ode_steps,
            dt=ode_dt,
            integrator=integrator,
            learnable_dt=learnable_dt,
            norm_between_steps=norm_between_steps,
        )
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        units = [mlp_in] + list(mlp_units) + [1]
        self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.pool_subset_k = pool_subset_k
        self.pool_subset_ratio = pool_subset_ratio
        self.pool_exclude_ends = pool_exclude_ends

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        node_repr = self.ode_block(edge_index, node_features, edge_features, training=training)
        node_pool, batch_pool = _sample_nodes_for_pooling(
            node_repr, batch_indices, training,
            self.pool_subset_k, self.pool_subset_ratio, self.pool_exclude_ends, True,
        )
        num_graphs = int(mx.max(batch_indices).item()) + 1
        graph_repr = scatter(node_pool, batch_pool, out_size=num_graphs, aggr="add")
        h = mx.concatenate([graph_repr, graph_features], axis=-1) if (
            self.rdkit_dim > 0 and graph_features is not None
        ) else graph_repr
        if training:
            h = self.dropout(h)
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < len(self.mlp_layers) - 1:
                h = nn.leaky_relu(h)
        return h


# --- ODE on top of DMPNN (velocity = one DMPNN step) ---
def _get_dmpnn():
    from mlx_graphs.nn.conv.dmpnn_conv import DMPNN
    return DMPNN


def _get_kadmpnn():
    from mlx_graphs.nn.conv.dmpnn_conv import KADMPNN
    return KADMPNN


class GraphODEBlockDMPNN(nn.Module):
    """ODE or additive skip: h += GNN(h). update_mode: ode (default) | additive.
    learnable_dt: if True, one Δt per step with Δt = 0.1 + 0.8*σ(α) ∈ [0.1, 0.9].
    norm_between_steps: 'layer' = LayerNorm after each step (DeeperGATGNN-style)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_steps: int = 4,
        dt: float = 0.25,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.update_mode = update_mode
        self.learnable_dt = learnable_dt
        self.norm_between_steps = norm_between_steps
        self.activation = activation
        self.act = nn.leaky_relu if activation == "leaky_relu" else nn.relu
        self.node_proj = Linear(node_dim, hidden_dim)
        if norm_between_steps == "layer":
            self.step_norm = nn.LayerNorm(hidden_dim)
        else:
            self.step_norm = None
        if learnable_dt:
            self.dt_alpha = mx.zeros((num_steps,))
        DMPNN = _get_dmpnn()
        self.dmpnn_one = DMPNN(
            node_dim=hidden_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            depth=1,
            dropout=0.0,
            activation=activation,
        )

    def __call__(
        self,
        directed_edge_index: mx.array,
        edge_reverse: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features)
        def step_fn(h_in: mx.array) -> mx.array:
            return self.dmpnn_one(
                directed_edge_index, edge_reverse, h_in, edge_features, training=training
            )
        norm_fn = (lambda h: self.step_norm(h)) if self.step_norm is not None else None
        if self.update_mode == "additive":
            h = _additive_integrate(h, self.num_steps, step_fn)
        elif self.learnable_dt:
            dts = _dt_from_alpha(self.dt_alpha)
            h = _ode_integrate_with_dts(h, dts, step_fn, self.integrator, norm_fn=norm_fn)
        else:
            h = _ode_integrate(h, self.dt, self.num_steps, step_fn, self.integrator, norm_fn=norm_fn)
        return self.act(h)


class GraphODEDMPNNRegressor(nn.Module):
    """ODE-DMPNN: velocity = one DMPNN step; needs edge_reverse. integrator: euler, heun, midpoint, rk4.
    mol_attention_steps: if > 0, use AttFP-style mol GRU/attention readout instead of sum pool.
    mlp_ode_steps: if > 0, use MLP-ODE (integrate in first MLP dim, same layer; only last proj to 1 is separate)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: list[int] = (128, 64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        mlp_ode_steps: int = 0,
        mlp_ode_dt: float = 0.25,
        mlp_ode_integrator: str = "euler",
        pool_subset_k: int = 0,
        pool_subset_ratio: float = 1.0,
        pool_exclude_ends: bool = False,
    ):
        super().__init__()
        self.mol_attention_steps = mol_attention_steps
        self.mlp_ode_steps = mlp_ode_steps
        self.ode_block = GraphODEBlockDMPNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_steps=ode_steps,
            dt=ode_dt,
            integrator=integrator,
            update_mode=update_mode,
            learnable_dt=learnable_dt,
            norm_between_steps=norm_between_steps,
        )
        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        if mlp_ode_steps > 0:
            first_dim = mlp_units[0] if mlp_units else 128
            self.mlp_ode = MLPODEBlock(
                mlp_in, first_dim, mlp_ode_steps, mlp_ode_dt, mlp_ode_integrator
            )
            self.mlp_layers = None
        else:
            self.mlp_ode = None
            units = [mlp_in] + list(mlp_units) + [1]
            self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.pool_subset_k = pool_subset_k
        self.pool_subset_ratio = pool_subset_ratio
        self.pool_exclude_ends = pool_exclude_ends

    def __call__(
        self,
        directed_edge_index: mx.array,
        edge_reverse: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        node_repr = self.ode_block(
            directed_edge_index, edge_reverse, node_features, edge_features, training=training
        )
        node_pool, batch_pool = _sample_nodes_for_pooling(
            node_repr, batch_indices, training,
            self.pool_subset_k, self.pool_subset_ratio, self.pool_exclude_ends, True,
        )
        if self.mol_readout is not None:
            graph_repr = self.mol_readout(node_pool, batch_pool, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(node_pool, batch_pool, out_size=num_graphs, aggr="add")
        h = mx.concatenate([graph_repr, graph_features], axis=-1) if (
            self.rdkit_dim > 0 and graph_features is not None
        ) else graph_repr
        if training:
            h = self.dropout(h)
        if self.mlp_ode is not None:
            return self.mlp_ode(h)
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < len(self.mlp_layers) - 1:
                h = nn.leaky_relu(h)
        return h


class GraphODEBlockKADMPNN(nn.Module):
    """ODE or additive skip with KADMPNN (one-step)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_steps: int = 4,
        dt: float = 0.25,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        kan_grid_size: int = 8,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.update_mode = update_mode
        self.learnable_dt = learnable_dt
        self.node_proj = Linear(node_dim, hidden_dim)
        if norm_between_steps == "layer":
            self.step_norm = nn.LayerNorm(hidden_dim)
        else:
            self.step_norm = None
        if learnable_dt:
            self.dt_alpha = mx.zeros((num_steps,))
        KADMPNN = _get_kadmpnn()
        self.kadmpnn_one = KADMPNN(
            node_dim=hidden_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            depth=1,
            dropout=0.0,
            kan_grid_size=kan_grid_size,
        )

    def __call__(
        self,
        directed_edge_index: mx.array,
        edge_reverse: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features)

        def step_fn(h_in: mx.array) -> mx.array:
            return self.kadmpnn_one(
                directed_edge_index, edge_reverse, h_in, edge_features, training=training
            )

        norm_fn = (lambda hh: self.step_norm(hh)) if self.step_norm is not None else None
        if self.update_mode == "additive":
            h = _additive_integrate(h, self.num_steps, step_fn)
        elif self.learnable_dt:
            dts = _dt_from_alpha(self.dt_alpha)
            h = _ode_integrate_with_dts(h, dts, step_fn, self.integrator, norm_fn=norm_fn)
        else:
            h = _ode_integrate(h, self.dt, self.num_steps, step_fn, self.integrator, norm_fn=norm_fn)
        return nn.relu(h)


class GraphODEKADMPNNRegressor(nn.Module):
    """ODE-KADMPNN regressor: KA-DMPNN inside ODE loop."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: list[int] = (128, 64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        mlp_ode_steps: int = 0,
        mlp_ode_dt: float = 0.25,
        mlp_ode_integrator: str = "euler",
        pool_subset_k: int = 0,
        pool_subset_ratio: float = 1.0,
        pool_exclude_ends: bool = False,
        kan_grid_size: int = 8,
    ):
        super().__init__()
        self.mlp_ode_steps = mlp_ode_steps
        self.ode_block = GraphODEBlockKADMPNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_steps=ode_steps,
            dt=ode_dt,
            integrator=integrator,
            update_mode=update_mode,
            learnable_dt=learnable_dt,
            norm_between_steps=norm_between_steps,
            kan_grid_size=kan_grid_size,
        )
        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        if mlp_ode_steps > 0:
            first_dim = mlp_units[0] if mlp_units else 128
            self.mlp_ode = MLPODEBlock(
                mlp_in, first_dim, mlp_ode_steps, mlp_ode_dt, mlp_ode_integrator
            )
            self.mlp_layers = None
        else:
            self.mlp_ode = None
            units = [mlp_in] + list(mlp_units) + [1]
            self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.pool_subset_k = pool_subset_k
        self.pool_subset_ratio = pool_subset_ratio
        self.pool_exclude_ends = pool_exclude_ends

    def __call__(
        self,
        directed_edge_index: mx.array,
        edge_reverse: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        node_repr = self.ode_block(
            directed_edge_index, edge_reverse, node_features, edge_features, training=training
        )
        node_pool, batch_pool = _sample_nodes_for_pooling(
            node_repr, batch_indices, training,
            self.pool_subset_k, self.pool_subset_ratio, self.pool_exclude_ends, True,
        )
        if self.mol_readout is not None:
            graph_repr = self.mol_readout(node_pool, batch_pool, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(node_pool, batch_pool, out_size=num_graphs, aggr="add")
        h = mx.concatenate([graph_repr, graph_features], axis=-1) if (
            self.rdkit_dim > 0 and graph_features is not None
        ) else graph_repr
        if training:
            h = self.dropout(h)
        if self.mlp_ode is not None:
            return self.mlp_ode(h)
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < len(self.mlp_layers) - 1:
                h = nn.leaky_relu(h)
        return h


# --- ODE on top of Graph Transformer (velocity = one block residual) ---
def _get_gt_block():
    from mlx_graphs.nn.conv.graph_transformer_conv import GraphTransformerBlock
    return GraphTransformerBlock


def _ode_gt_activation_fn(name: str):
    """Return activation callable: relu, leaky_relu, or gelu."""
    if name == "leaky_relu":
        return nn.leaky_relu
    if name == "gelu":
        return nn.gelu
    return nn.relu


class GraphODEBlockGT(nn.Module):
    """ODE or additive skip: h += GNN(h). update_mode: ode (default) | additive.
    learnable_dt: if True, one Δt per step with Δt = 0.1 + 0.8*σ(α) ∈ [0.1, 0.9].
    norm_between_steps: 'layer' = LayerNorm after each step (DeeperGATGNN-style).
    ffn_dim: FFN hidden dim (default None = 4*hidden_dim). Use hidden_dim for same-dim (no 4x expansion).
    activation: relu (default), leaky_relu, or gelu — applied after ODE integration."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_steps: int = 4,
        dt: float = 0.25,
        dropout: float = 0.1,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        use_edge_features: bool = True,
        ffn_dim: Optional[int] = None,
        activation: str = "relu",
    ):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.update_mode = update_mode
        self.learnable_dt = learnable_dt
        self.norm_between_steps = norm_between_steps
        self.node_proj = Linear(node_dim, hidden_dim)
        if norm_between_steps == "layer":
            self.step_norm = nn.LayerNorm(hidden_dim)
        else:
            self.step_norm = None
        if learnable_dt:
            self.dt_alpha = mx.zeros((num_steps,))
        GraphTransformerBlock = _get_gt_block()
        self.gt_block = GraphTransformerBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            ffn_dim=ffn_dim,
            use_edge_features=use_edge_features,
        )
        self.act = _ode_gt_activation_fn(activation)

    def __call__(
        self,
        node_features: mx.array,
        batch_indices: mx.array,
        edge_index: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features)
        def step_fn(h_in: mx.array) -> mx.array:
            return self.gt_block(h_in, batch_indices, edge_index, edge_features, training=training)
        norm_fn = (lambda h: self.step_norm(h)) if self.step_norm is not None else None
        if self.update_mode == "additive":
            h = _additive_integrate(h, self.num_steps, step_fn)
        elif self.learnable_dt:
            dts = _dt_from_alpha(self.dt_alpha)
            h = _ode_integrate_with_dts(h, dts, step_fn, self.integrator, norm_fn=norm_fn)
        else:
            h = _ode_integrate(h, self.dt, self.num_steps, step_fn, self.integrator, norm_fn=norm_fn)
        return self.act(h)


class GraphODEGTRegressor(nn.Module):
    """ODE-GT or additive skip. mlp_ode_steps: if > 0, use MLP-ODE.
    ffn_dim: FFN hidden dim (default None = 4*hidden_dim). Use hidden_dim for same-dim (no 4x expansion).
    activation: relu (default), leaky_relu, or gelu — applied after ODE integration."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: list[int] = (128, 64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        mlp_ode_steps: int = 0,
        mlp_ode_dt: float = 0.25,
        mlp_ode_integrator: str = "euler",
        pool_subset_k: int = 0,
        pool_subset_ratio: float = 1.0,
        pool_exclude_ends: bool = False,
        use_edge_features: bool = True,
        ffn_dim: Optional[int] = None,
        activation: str = "relu",
    ):
        super().__init__()
        self.mlp_ode_steps = mlp_ode_steps
        self.ode_block = GraphODEBlockGT(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_steps=ode_steps,
            dt=ode_dt,
            dropout=dropout,
            integrator=integrator,
            update_mode=update_mode,
            learnable_dt=learnable_dt,
            norm_between_steps=norm_between_steps,
            use_edge_features=use_edge_features,
            ffn_dim=ffn_dim,
            activation=activation,
        )
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        if mlp_ode_steps > 0:
            first_dim = mlp_units[0] if mlp_units else 128
            self.mlp_ode = MLPODEBlock(mlp_in, first_dim, mlp_ode_steps, mlp_ode_dt, mlp_ode_integrator)
            self.mlp_layers = None
        else:
            self.mlp_ode = None
            units = [mlp_in] + list(mlp_units) + [1]
            self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.pool_subset_k = pool_subset_k
        self.pool_subset_ratio = pool_subset_ratio
        self.pool_exclude_ends = pool_exclude_ends

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        node_repr = self.ode_block(
            node_features, batch_indices, edge_index, edge_features, training=training
        )
        node_pool, batch_pool = _sample_nodes_for_pooling(
            node_repr, batch_indices, training,
            self.pool_subset_k, self.pool_subset_ratio, self.pool_exclude_ends, True,
        )
        num_graphs = int(mx.max(batch_indices).item()) + 1
        graph_repr = scatter(node_pool, batch_pool, out_size=num_graphs, aggr="add")
        h = mx.concatenate([graph_repr, graph_features], axis=-1) if (
            self.rdkit_dim > 0 and graph_features is not None
        ) else graph_repr
        if training:
            h = self.dropout(h)
        if self.mlp_ode is not None:
            return self.mlp_ode(h)
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < len(self.mlp_layers) - 1:
                h = nn.leaky_relu(h)
        return h


# --- ODE on top of GAT (velocity = one GAT layer) ---
def _get_gat_conv():
    from mlx_graphs.nn.conv.gat_conv import GATConv
    return GATConv


def _get_gatv2_conv():
    from mlx_graphs.nn.conv.gatv2_conv import GATv2Conv
    return GATv2Conv


def _get_gcn_conv():
    from mlx_graphs.nn.conv.gcn_conv import GCNConv
    return GCNConv


def _get_kan_linear():
    from mlx_graphs.nn.conv.ka_gnn_conv import KANLinear
    return KANLinear


class GraphODEBlockGCN(nn.Module):
    """ODE or additive skip with GCN: h += GNN(h)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_steps: int = 4,
        dt: float = 0.25,
        dropout: float = 0.1,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
    ):
        del edge_dim, dropout  # GCN does not use edge features/dropout in conv call.
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.update_mode = update_mode
        self.learnable_dt = learnable_dt
        self.norm_between_steps = norm_between_steps
        self.node_proj = Linear(node_dim, hidden_dim)
        if norm_between_steps == "layer":
            self.step_norm = nn.LayerNorm(hidden_dim)
        else:
            self.step_norm = None
        if learnable_dt:
            self.dt_alpha = mx.zeros((num_steps,))
        GCNConv = _get_gcn_conv()
        self.gcn_one = GCNConv(hidden_dim, hidden_dim, add_self_loops=True)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        del edge_features, training
        h = self.node_proj(node_features)

        def step_fn(h_in: mx.array) -> mx.array:
            return nn.relu(self.gcn_one(edge_index, h_in))

        norm_fn = (lambda hh: self.step_norm(hh)) if self.step_norm is not None else None
        if self.update_mode == "additive":
            h = _additive_integrate(h, self.num_steps, step_fn)
        elif self.learnable_dt:
            dts = _dt_from_alpha(self.dt_alpha)
            h = _ode_integrate_with_dts(h, dts, step_fn, self.integrator, norm_fn=norm_fn)
        else:
            h = _ode_integrate(h, self.dt, self.num_steps, step_fn, self.integrator, norm_fn=norm_fn)
        return nn.relu(h)


class GraphODEGCNRegressor(nn.Module):
    """ODE-GCN or additive skip. Supports optional mol attention and MLP-ODE head."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: list[int] = (128, 64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        mlp_ode_steps: int = 0,
        mlp_ode_dt: float = 0.25,
        mlp_ode_integrator: str = "euler",
        pool_subset_k: int = 0,
        pool_subset_ratio: float = 1.0,
        pool_exclude_ends: bool = False,
    ):
        super().__init__()
        self.mlp_ode_steps = mlp_ode_steps
        self.ode_block = GraphODEBlockGCN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_steps=ode_steps,
            dt=ode_dt,
            dropout=dropout,
            integrator=integrator,
            update_mode=update_mode,
            learnable_dt=learnable_dt,
            norm_between_steps=norm_between_steps,
        )
        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        if mlp_ode_steps > 0:
            first_dim = mlp_units[0] if mlp_units else 128
            self.mlp_ode = MLPODEBlock(mlp_in, first_dim, mlp_ode_steps, mlp_ode_dt, mlp_ode_integrator)
            self.mlp_layers = None
        else:
            self.mlp_ode = None
            units = [mlp_in] + list(mlp_units) + [1]
            self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.pool_subset_k = pool_subset_k
        self.pool_subset_ratio = pool_subset_ratio
        self.pool_exclude_ends = pool_exclude_ends

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        node_repr = self.ode_block(
            edge_index, node_features, edge_features, training=training
        )
        node_pool, batch_pool = _sample_nodes_for_pooling(
            node_repr, batch_indices, training,
            self.pool_subset_k, self.pool_subset_ratio, self.pool_exclude_ends, True,
        )
        if self.mol_readout is not None:
            graph_repr = self.mol_readout(node_pool, batch_pool, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(node_pool, batch_pool, out_size=num_graphs, aggr="add")
        h = mx.concatenate([graph_repr, graph_features], axis=-1) if (
            self.rdkit_dim > 0 and graph_features is not None
        ) else graph_repr
        if training:
            h = self.dropout(h)
        if self.mlp_ode is not None:
            return self.mlp_ode(h)
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < len(self.mlp_layers) - 1:
                h = nn.leaky_relu(h)
        return h


class GraphODEBlockKAGCN(nn.Module):
    """ODE or additive skip with KA-GCN step."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_steps: int = 4,
        dt: float = 0.25,
        dropout: float = 0.1,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        kan_grid_size: int = 8,
    ):
        del edge_dim, dropout
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.update_mode = update_mode
        self.learnable_dt = learnable_dt
        self.node_proj = Linear(node_dim, hidden_dim)
        if norm_between_steps == "layer":
            self.step_norm = nn.LayerNorm(hidden_dim)
        else:
            self.step_norm = None
        if learnable_dt:
            self.dt_alpha = mx.zeros((num_steps,))
        GCNConv = _get_gcn_conv()
        KANLinear = _get_kan_linear()
        self.gcn_one = GCNConv(hidden_dim, hidden_dim, add_self_loops=True)
        self.kan_one = KANLinear(hidden_dim, hidden_dim, grid_size=kan_grid_size, bias=True)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        del edge_features, training
        h = self.node_proj(node_features)

        def step_fn(h_in: mx.array) -> mx.array:
            msg = self.gcn_one(edge_index, h_in)
            return nn.relu(self.kan_one(msg))

        norm_fn = (lambda hh: self.step_norm(hh)) if self.step_norm is not None else None
        if self.update_mode == "additive":
            h = _additive_integrate(h, self.num_steps, step_fn)
        elif self.learnable_dt:
            dts = _dt_from_alpha(self.dt_alpha)
            h = _ode_integrate_with_dts(h, dts, step_fn, self.integrator, norm_fn=norm_fn)
        else:
            h = _ode_integrate(h, self.dt, self.num_steps, step_fn, self.integrator, norm_fn=norm_fn)
        return nn.relu(h)


class GraphODEKAGCNRegressor(nn.Module):
    """ODE-KAGCN or additive skip. Supports optional mol attention and MLP-ODE head."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: list[int] = (128, 64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        mlp_ode_steps: int = 0,
        mlp_ode_dt: float = 0.25,
        mlp_ode_integrator: str = "euler",
        pool_subset_k: int = 0,
        pool_subset_ratio: float = 1.0,
        pool_exclude_ends: bool = False,
        kan_grid_size: int = 8,
    ):
        super().__init__()
        self.mlp_ode_steps = mlp_ode_steps
        self.ode_block = GraphODEBlockKAGCN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_steps=ode_steps,
            dt=ode_dt,
            dropout=dropout,
            integrator=integrator,
            update_mode=update_mode,
            learnable_dt=learnable_dt,
            norm_between_steps=norm_between_steps,
            kan_grid_size=kan_grid_size,
        )
        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        if mlp_ode_steps > 0:
            first_dim = mlp_units[0] if mlp_units else 128
            self.mlp_ode = MLPODEBlock(mlp_in, first_dim, mlp_ode_steps, mlp_ode_dt, mlp_ode_integrator)
            self.mlp_layers = None
        else:
            self.mlp_ode = None
            units = [mlp_in] + list(mlp_units) + [1]
            self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.pool_subset_k = pool_subset_k
        self.pool_subset_ratio = pool_subset_ratio
        self.pool_exclude_ends = pool_exclude_ends

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        node_repr = self.ode_block(
            edge_index, node_features, edge_features, training=training
        )
        node_pool, batch_pool = _sample_nodes_for_pooling(
            node_repr, batch_indices, training,
            self.pool_subset_k, self.pool_subset_ratio, self.pool_exclude_ends, True,
        )
        if self.mol_readout is not None:
            graph_repr = self.mol_readout(node_pool, batch_pool, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(node_pool, batch_pool, out_size=num_graphs, aggr="add")
        h = mx.concatenate([graph_repr, graph_features], axis=-1) if (
            self.rdkit_dim > 0 and graph_features is not None
        ) else graph_repr
        if training:
            h = self.dropout(h)
        if self.mlp_ode is not None:
            return self.mlp_ode(h)
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < len(self.mlp_layers) - 1:
                h = nn.leaky_relu(h)
        return h


class GraphODEBlockKAGAT(nn.Module):
    """ODE or additive skip with KA-GAT step."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        num_steps: int = 4,
        dt: float = 0.25,
        dropout: float = 0.1,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        kan_grid_size: int = 8,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.update_mode = update_mode
        self.learnable_dt = learnable_dt
        self.node_proj = Linear(node_dim, hidden_dim)
        if norm_between_steps == "layer":
            self.step_norm = nn.LayerNorm(hidden_dim)
        else:
            self.step_norm = None
        if learnable_dt:
            self.dt_alpha = mx.zeros((num_steps,))
        GATConv = _get_gat_conv()
        KANLinear = _get_kan_linear()
        self.gat_one = GATConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_features_dim=edge_dim,
        )
        self.kan_one = KANLinear(hidden_dim, hidden_dim, grid_size=kan_grid_size, bias=True)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features)

        def step_fn(h_in: mx.array) -> mx.array:
            msg = self.gat_one(edge_index, h_in, edge_features)
            return nn.relu(self.kan_one(msg))

        norm_fn = (lambda hh: self.step_norm(hh)) if self.step_norm is not None else None
        if self.update_mode == "additive":
            h = _additive_integrate(h, self.num_steps, step_fn)
        elif self.learnable_dt:
            dts = _dt_from_alpha(self.dt_alpha)
            h = _ode_integrate_with_dts(h, dts, step_fn, self.integrator, norm_fn=norm_fn)
        else:
            h = _ode_integrate(h, self.dt, self.num_steps, step_fn, self.integrator, norm_fn=norm_fn)
        return nn.relu(h)


class GraphODEKAGATRegressor(nn.Module):
    """ODE-KAGAT or additive skip. Supports optional mol attention and MLP-ODE head."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: list[int] = (128, 64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        mlp_ode_steps: int = 0,
        mlp_ode_dt: float = 0.25,
        mlp_ode_integrator: str = "euler",
        pool_subset_k: int = 0,
        pool_subset_ratio: float = 1.0,
        pool_exclude_ends: bool = False,
        kan_grid_size: int = 8,
    ):
        super().__init__()
        self.mlp_ode_steps = mlp_ode_steps
        self.ode_block = GraphODEBlockKAGAT(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            num_steps=ode_steps,
            dt=ode_dt,
            dropout=dropout,
            integrator=integrator,
            update_mode=update_mode,
            learnable_dt=learnable_dt,
            norm_between_steps=norm_between_steps,
            kan_grid_size=kan_grid_size,
        )
        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        if mlp_ode_steps > 0:
            first_dim = mlp_units[0] if mlp_units else 128
            self.mlp_ode = MLPODEBlock(mlp_in, first_dim, mlp_ode_steps, mlp_ode_dt, mlp_ode_integrator)
            self.mlp_layers = None
        else:
            self.mlp_ode = None
            units = [mlp_in] + list(mlp_units) + [1]
            self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.pool_subset_k = pool_subset_k
        self.pool_subset_ratio = pool_subset_ratio
        self.pool_exclude_ends = pool_exclude_ends

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        node_repr = self.ode_block(
            edge_index, node_features, edge_features, training=training
        )
        node_pool, batch_pool = _sample_nodes_for_pooling(
            node_repr, batch_indices, training,
            self.pool_subset_k, self.pool_subset_ratio, self.pool_exclude_ends, True,
        )
        if self.mol_readout is not None:
            graph_repr = self.mol_readout(node_pool, batch_pool, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(node_pool, batch_pool, out_size=num_graphs, aggr="add")
        h = mx.concatenate([graph_repr, graph_features], axis=-1) if (
            self.rdkit_dim > 0 and graph_features is not None
        ) else graph_repr
        if training:
            h = self.dropout(h)
        if self.mlp_ode is not None:
            return self.mlp_ode(h)
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < len(self.mlp_layers) - 1:
                h = nn.leaky_relu(h)
        return h


class GraphODEBlockGAT(nn.Module):
    """ODE or additive skip: h += GNN(h). update_mode: ode (default) | additive.
    learnable_dt: if True, one Δt per step with Δt = 0.1 + 0.8*σ(α) ∈ [0.1, 0.9].
    norm_between_steps: 'layer' = LayerNorm after each step (DeeperGATGNN-style)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        num_steps: int = 4,
        dt: float = 0.25,
        dropout: float = 0.1,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
    ):
        super().__init__()
        assert hidden_dim % heads == 0
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.update_mode = update_mode
        self.learnable_dt = learnable_dt
        self.norm_between_steps = norm_between_steps
        self.node_proj = Linear(node_dim, hidden_dim)
        if norm_between_steps == "layer":
            self.step_norm = nn.LayerNorm(hidden_dim)
        else:
            self.step_norm = None
        if learnable_dt:
            self.dt_alpha = mx.zeros((num_steps,))
        GATConv = _get_gat_conv()
        self.gat_one = GATConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_features_dim=edge_dim,
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features)
        def step_fn(h_in: mx.array) -> mx.array:
            return nn.leaky_relu(self.gat_one(edge_index, h_in, edge_features))
        norm_fn = (lambda h: self.step_norm(h)) if self.step_norm is not None else None
        if self.update_mode == "additive":
            h = _additive_integrate(h, self.num_steps, step_fn)
        elif self.learnable_dt:
            dts = _dt_from_alpha(self.dt_alpha)
            h = _ode_integrate_with_dts(h, dts, step_fn, self.integrator, norm_fn=norm_fn)
        else:
            h = _ode_integrate(h, self.dt, self.num_steps, step_fn, self.integrator, norm_fn=norm_fn)
        return nn.relu(h)


class GraphODEGATRegressor(nn.Module):
    """ODE-GAT or additive skip. update_mode: ode | additive.
    mol_attention_steps: if > 0, use AttFP-style mol readout. mlp_ode_steps: if > 0, use MLP-ODE."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: list[int] = (128, 64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        mlp_ode_steps: int = 0,
        mlp_ode_dt: float = 0.25,
        mlp_ode_integrator: str = "euler",
        pool_subset_k: int = 0,
        pool_subset_ratio: float = 1.0,
        pool_exclude_ends: bool = False,
    ):
        super().__init__()
        self.mol_attention_steps = mol_attention_steps
        self.mlp_ode_steps = mlp_ode_steps
        self.ode_block = GraphODEBlockGAT(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            num_steps=ode_steps,
            dt=ode_dt,
            dropout=dropout,
            integrator=integrator,
            update_mode=update_mode,
            learnable_dt=learnable_dt,
            norm_between_steps=norm_between_steps,
        )
        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        if mlp_ode_steps > 0:
            first_dim = mlp_units[0] if mlp_units else 128
            self.mlp_ode = MLPODEBlock(mlp_in, first_dim, mlp_ode_steps, mlp_ode_dt, mlp_ode_integrator)
            self.mlp_layers = None
        else:
            self.mlp_ode = None
            units = [mlp_in] + list(mlp_units) + [1]
            self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.pool_subset_k = pool_subset_k
        self.pool_subset_ratio = pool_subset_ratio
        self.pool_exclude_ends = pool_exclude_ends

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        node_repr = self.ode_block(
            edge_index, node_features, edge_features, training=training
        )
        node_pool, batch_pool = _sample_nodes_for_pooling(
            node_repr, batch_indices, training,
            self.pool_subset_k, self.pool_subset_ratio, self.pool_exclude_ends, True,
        )
        if self.mol_readout is not None:
            graph_repr = self.mol_readout(node_pool, batch_pool, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(node_pool, batch_pool, out_size=num_graphs, aggr="add")
        h = mx.concatenate([graph_repr, graph_features], axis=-1) if (
            self.rdkit_dim > 0 and graph_features is not None
        ) else graph_repr
        if training:
            h = self.dropout(h)
        if self.mlp_ode is not None:
            return self.mlp_ode(h)
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < len(self.mlp_layers) - 1:
                h = nn.leaky_relu(h)
        return h


class GraphODEBlockGATv2(nn.Module):
    """ODE or additive skip with GATv2: h += GNN(h). Same as GraphODEBlockGAT but uses GATv2Conv."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        num_steps: int = 4,
        dt: float = 0.25,
        dropout: float = 0.1,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
    ):
        super().__init__()
        assert hidden_dim % heads == 0
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.update_mode = update_mode
        self.learnable_dt = learnable_dt
        self.norm_between_steps = norm_between_steps
        self.node_proj = Linear(node_dim, hidden_dim)
        if norm_between_steps == "layer":
            self.step_norm = nn.LayerNorm(hidden_dim)
        else:
            self.step_norm = None
        if learnable_dt:
            self.dt_alpha = mx.zeros((num_steps,))
        GATv2Conv = _get_gatv2_conv()
        self.gatv2_one = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_features_dim=edge_dim,
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features)
        def step_fn(h_in: mx.array) -> mx.array:
            return nn.leaky_relu(self.gatv2_one(edge_index, h_in, edge_features))
        norm_fn = (lambda h: self.step_norm(h)) if self.step_norm is not None else None
        if self.update_mode == "additive":
            h = _additive_integrate(h, self.num_steps, step_fn)
        elif self.learnable_dt:
            dts = _dt_from_alpha(self.dt_alpha)
            h = _ode_integrate_with_dts(h, dts, step_fn, self.integrator, norm_fn=norm_fn)
        else:
            h = _ode_integrate(h, self.dt, self.num_steps, step_fn, self.integrator, norm_fn=norm_fn)
        return nn.relu(h)


class GraphODEGATv2Regressor(nn.Module):
    """ODE-GATv2 or additive skip. mol_attention_steps: if > 0, use AttFP-style mol readout. mlp_ode_steps: if > 0, use MLP-ODE."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: list[int] = (128, 64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        mlp_ode_steps: int = 0,
        mlp_ode_dt: float = 0.25,
        mlp_ode_integrator: str = "euler",
        pool_subset_k: int = 0,
        pool_subset_ratio: float = 1.0,
        pool_exclude_ends: bool = False,
    ):
        super().__init__()
        self.mol_attention_steps = mol_attention_steps
        self.mlp_ode_steps = mlp_ode_steps
        self.ode_block = GraphODEBlockGATv2(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            num_steps=ode_steps,
            dt=ode_dt,
            dropout=dropout,
            integrator=integrator,
            update_mode=update_mode,
            learnable_dt=learnable_dt,
            norm_between_steps=norm_between_steps,
        )
        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        if mlp_ode_steps > 0:
            first_dim = mlp_units[0] if mlp_units else 128
            self.mlp_ode = MLPODEBlock(mlp_in, first_dim, mlp_ode_steps, mlp_ode_dt, mlp_ode_integrator)
            self.mlp_layers = None
        else:
            self.mlp_ode = None
            units = [mlp_in] + list(mlp_units) + [1]
            self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.pool_subset_k = pool_subset_k
        self.pool_subset_ratio = pool_subset_ratio
        self.pool_exclude_ends = pool_exclude_ends

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        node_repr = self.ode_block(
            edge_index, node_features, edge_features, training=training
        )
        node_pool, batch_pool = _sample_nodes_for_pooling(
            node_repr, batch_indices, training,
            self.pool_subset_k, self.pool_subset_ratio, self.pool_exclude_ends, True,
        )
        if self.mol_readout is not None:
            graph_repr = self.mol_readout(node_pool, batch_pool, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(node_pool, batch_pool, out_size=num_graphs, aggr="add")
        h = mx.concatenate([graph_repr, graph_features], axis=-1) if (
            self.rdkit_dim > 0 and graph_features is not None
        ) else graph_repr
        if training:
            h = self.dropout(h)
        if self.mlp_ode is not None:
            return self.mlp_ode(h)
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < len(self.mlp_layers) - 1:
                h = nn.leaky_relu(h)
        return h


# --- ODE on top of GINE (velocity = one GINE layer) ---
def _get_gin_conv():
    from mlx_graphs.nn.conv.gin_conv import GINConv
    return GINConv


class GraphODEBlockGINE(nn.Module):
    """ODE or additive skip: h += GNN(h). update_mode: ode (default) | additive.
    learnable_dt: if True, one Δt per step with Δt = 0.1 + 0.8*σ(α) ∈ [0.1, 0.9].
    norm_between_steps: 'layer' = LayerNorm after each step (DeeperGATGNN-style)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_steps: int = 4,
        dt: float = 0.25,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.update_mode = update_mode
        self.learnable_dt = learnable_dt
        self.norm_between_steps = norm_between_steps
        self.node_proj = Linear(node_dim, hidden_dim)
        if norm_between_steps == "layer":
            self.step_norm = nn.LayerNorm(hidden_dim)
        else:
            self.step_norm = None
        if learnable_dt:
            self.dt_alpha = mx.zeros((num_steps,))
        GINConv = _get_gin_conv()
        gine_mlp = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.gine_one = GINConv(
            gine_mlp,
            edge_features_dim=edge_dim,
            node_features_dim=hidden_dim,
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features)
        def step_fn(h_in: mx.array) -> mx.array:
            return nn.relu(self.gine_one(edge_index, h_in, edge_features))
        norm_fn = (lambda h: self.step_norm(h)) if self.step_norm is not None else None
        if self.update_mode == "additive":
            h = _additive_integrate(h, self.num_steps, step_fn)
        elif self.learnable_dt:
            dts = _dt_from_alpha(self.dt_alpha)
            h = _ode_integrate_with_dts(h, dts, step_fn, self.integrator, norm_fn=norm_fn)
        else:
            h = _ode_integrate(h, self.dt, self.num_steps, step_fn, self.integrator, norm_fn=norm_fn)
        return nn.relu(h)


class GraphODEGINERegressor(nn.Module):
    """ODE-GINE or additive skip. mol_attention_steps: if > 0, use AttFP-style mol readout. mlp_ode_steps: if > 0, use MLP-ODE."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: list[int] = (128, 64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        mlp_ode_steps: int = 0,
        mlp_ode_dt: float = 0.25,
        mlp_ode_integrator: str = "euler",
        pool_subset_k: int = 0,
        pool_subset_ratio: float = 1.0,
        pool_exclude_ends: bool = False,
    ):
        super().__init__()
        self.mol_attention_steps = mol_attention_steps
        self.mlp_ode_steps = mlp_ode_steps
        self.ode_block = GraphODEBlockGINE(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_steps=ode_steps,
            dt=ode_dt,
            integrator=integrator,
            update_mode=update_mode,
            learnable_dt=learnable_dt,
            norm_between_steps=norm_between_steps,
        )
        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        if mlp_ode_steps > 0:
            first_dim = mlp_units[0] if mlp_units else 128
            self.mlp_ode = MLPODEBlock(mlp_in, first_dim, mlp_ode_steps, mlp_ode_dt, mlp_ode_integrator)
            self.mlp_layers = None
        else:
            self.mlp_ode = None
            units = [mlp_in] + list(mlp_units) + [1]
            self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.pool_subset_k = pool_subset_k
        self.pool_subset_ratio = pool_subset_ratio
        self.pool_exclude_ends = pool_exclude_ends

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        node_repr = self.ode_block(
            edge_index, node_features, edge_features, training=training
        )
        node_pool, batch_pool = _sample_nodes_for_pooling(
            node_repr, batch_indices, training,
            self.pool_subset_k, self.pool_subset_ratio, self.pool_exclude_ends, self.mol_readout is None,
        )
        if self.mol_readout is not None:
            graph_repr = self.mol_readout(node_pool, batch_pool, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(node_pool, batch_pool, out_size=num_graphs, aggr="add")
        h = mx.concatenate([graph_repr, graph_features], axis=-1) if (
            self.rdkit_dim > 0 and graph_features is not None
        ) else graph_repr
        if training:
            h = self.dropout(h)
        if self.mlp_ode is not None:
            return self.mlp_ode(h)
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < len(self.mlp_layers) - 1:
                h = nn.leaky_relu(h)
        return h


# --- AttentiveFP simplification for ODE (optional mol-level) ---
# Default AttentiveFP uses multiple atom-level blocks (radius) and multiple mol-level blocks (T).
# We use a single AttentiveFP atom-level step with ODE, then optionally the same mol-level as AttentiveFP.
def _get_attfp_node_step():
    from mlx_graphs.nn.conv.attentivefp_conv import AttentiveFPNodeStep
    return AttentiveFPNodeStep


def _get_attfp_gru_cell():
    from mlx_graphs.nn.conv.attentivefp_conv import _GRUCell
    return _GRUCell


class GraphODEBlockAttFP(nn.Module):
    """ODE or additive skip over a single AttentiveFP atom-level step (no mol-level).
    update_mode: ode (default) | additive. learnable_dt: one Δt per step in [0.1, 0.9].
    norm_between_steps: 'layer' = LayerNorm after each step (DeeperGATGNN-style)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_steps: int = 4,
        dt: float = 0.25,
        dropout: float = 0.1,
        integrator: str = "euler",
        update_mode: str = "ode",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.integrator = integrator
        self.update_mode = update_mode
        self.learnable_dt = learnable_dt
        self.norm_between_steps = norm_between_steps
        self.node_proj = Linear(node_dim, hidden_dim)
        if norm_between_steps == "layer":
            self.step_norm = nn.LayerNorm(hidden_dim)
        else:
            self.step_norm = None
        if learnable_dt:
            self.dt_alpha = mx.zeros((num_steps,))
        AttentiveFPNodeStep = _get_attfp_node_step()
        self.attfp_step = AttentiveFPNodeStep(
            fp_dim=hidden_dim,
            edge_dim=edge_dim,
            dropout=dropout,
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features)
        def step_fn(h_in: mx.array) -> mx.array:
            return self.attfp_step(edge_index, h_in, edge_features, training=training)
        if self.update_mode == "additive":
            h = _additive_integrate(h, self.num_steps, step_fn)
        elif self.learnable_dt:
            dts = _dt_from_alpha(self.dt_alpha)
            h = _ode_integrate_with_dts(h, dts, step_fn, self.integrator)
        else:
            h = _ode_integrate(h, self.dt, self.num_steps, step_fn, self.integrator)
        return nn.relu(h)


class GraphODEAttFPRegressor(nn.Module):
    """ODE-AttFP: single atom-level block with ODE, optional mol-level (T steps) like AttentiveFP.
    update_mode: ode | additive. mol_steps=0: no mol-level (only pool + MLP). mol_steps>0: mol-level GRU refinement."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: list[int] = (64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        update_mode: str = "ode",
        mol_steps: int = 2,
        mol_integrator: str = "euler",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
        pool_subset_k: int = 0,
        pool_subset_ratio: float = 1.0,
        pool_exclude_ends: bool = False,
    ):
        super().__init__()
        self.mol_steps = mol_steps
        if mol_integrator not in ("euler", "heun"):
            raise ValueError("mol_integrator must be one of ('euler', 'heun')")
        self.mol_integrator = mol_integrator
        self.ode_block = GraphODEBlockAttFP(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_steps=ode_steps,
            dt=ode_dt,
            dropout=dropout,
            integrator=integrator,
            update_mode=update_mode,
            learnable_dt=learnable_dt,
            norm_between_steps=norm_between_steps,
        )
        if mol_steps > 0:
            _GRUCell = _get_attfp_gru_cell()
            self.mol_align = Linear(2 * hidden_dim, 1)
            self.mol_attend = Linear(hidden_dim, hidden_dim)
            self.mol_gru = _GRUCell(hidden_dim, hidden_dim)
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        units = [mlp_in] + list(mlp_units) + [1]
        self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.pool_subset_k = pool_subset_k
        self.pool_subset_ratio = pool_subset_ratio
        self.pool_exclude_ends = pool_exclude_ends

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        atom_feature = self.ode_block(
            edge_index, node_features, edge_features, training=training
        )
        atom_pool, batch_pool = _sample_nodes_for_pooling(
            atom_feature, batch_indices, training,
            self.pool_subset_k, self.pool_subset_ratio, self.pool_exclude_ends, True,
        )
        num_graphs = int(mx.max(batch_indices).item()) + 1
        if self.mol_steps > 0:
            mol_feature = scatter(
                atom_pool,
                batch_pool,
                out_size=num_graphs,
                aggr="add",
            )
            mol_feature = nn.relu(mol_feature)  # match OpenDrugAI/AttentiveFP AttentiveLayers.py
            def _mol_context(mol_h: mx.array) -> mx.array:
                mol_expand = mol_h[batch_pool]
                align_inp = mx.concatenate([mol_expand, atom_pool], axis=-1)
                align_score = nn.leaky_relu(self.mol_align(align_inp)).reshape(-1)
                max_per_graph = scatter(
                    align_score, batch_pool, out_size=num_graphs, aggr="max"
                )
                attention_weight = mx.exp(align_score - max_per_graph[batch_pool])
                norm = scatter(
                    attention_weight,
                    batch_pool,
                    out_size=num_graphs,
                    aggr="add",
                )
                attention_weight = attention_weight / (norm[batch_pool] + 1e-8)
                atom_transform = self.mol_attend(
                    self.dropout(atom_pool) if training else atom_pool
                )
                mol_ctx = scatter(
                    attention_weight[:, None] * atom_transform,
                    batch_pool,
                    out_size=num_graphs,
                    aggr="add",
                )
                return nn.elu(mol_ctx)

            def _mol_update(mol_h: mx.array, mol_ctx: mx.array) -> mx.array:
                return nn.relu(self.mol_gru(mol_h, mol_ctx))

            for _ in range(self.mol_steps):
                mol_context = _mol_context(mol_feature)
                if self.mol_integrator == "euler":
                    mol_feature = _mol_update(mol_feature, mol_context)
                else:
                    pred = _mol_update(mol_feature, mol_context)
                    mol_context_2 = _mol_context(pred)
                    corr = _mol_update(pred, mol_context_2)
                    mol_feature = 0.5 * (corr + mol_feature)
            graph_repr = mol_feature
        else:
            graph_repr = scatter(
                atom_pool, batch_pool, out_size=num_graphs, aggr="add"
            )
        h = mx.concatenate([graph_repr, graph_features], axis=-1) if (
            self.rdkit_dim > 0 and graph_features is not None
        ) else graph_repr
        if training:
            h = self.dropout(h)
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < len(self.mlp_layers) - 1:
                h = nn.leaky_relu(h)
        return h
