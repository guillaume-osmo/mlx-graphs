from __future__ import annotations

from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv.gat_conv import GATConv
from mlx_graphs.nn.conv.gcn_conv import GCNConv
from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.mol_attention_readout import MolAttentionReadout
from mlx_graphs.utils import scatter


class KANLinear(nn.Module):
    """Lightweight KAN-inspired linear layer.

    This is a practical approximation for MLX: a standard linear path plus
    learnable bounded radial basis terms over fixed scalar knots per input channel.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        grid_size: int = 8,
        knot_min: float = -2.0,
        knot_max: float = 2.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        self.base = Linear(input_dims, output_dims, bias=bias)
        self.knots = mx.linspace(knot_min, knot_max, grid_size)
        # Start from near-identity (linear-only) for training stability.
        self.spline_weight = mx.zeros((output_dims, input_dims, grid_size))
        self.spline_scale = mx.array([0.05], dtype=mx.float32)
        knot_step = float(knot_max - knot_min) / float(grid_size - 1)
        sigma = max(1e-3, 0.5 * knot_step)
        self._inv_var = 1.0 / (2.0 * sigma * sigma)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, in_dim), basis: (N, in_dim, grid_size)
        # Clamp dynamic range so high-magnitude activations do not explode basis responses.
        x_safe = 3.0 * mx.tanh(x / 3.0)
        delta = mx.expand_dims(x_safe, axis=-1) - self.knots
        basis = mx.exp(-(delta * delta) * self._inv_var)
        spline = mx.einsum("nig,oig->no", basis, self.spline_weight)
        return self.base(x) + self.spline_scale[0] * spline


class KAGCNRegressor(nn.Module):
    """KA-GCN style regressor for molecular property prediction.
    residual_dt: None = replace h=relu(kan(gcn(h))); "add" = relu(norm(h+y)); 0.5 = h+0.5*(y-h)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
        mlp_units: tuple[int, ...] = (64, 32),
        rdkit_dim: int = 0,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        kan_grid_size: int = 8,
        residual_dt: Optional[Union[float, str]] = None,
    ):
        del edge_dim  # GCN variant does not use edge features here.
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.residual_dt = residual_dt
        self.rdkit_dim = rdkit_dim
        self.node_proj = Linear(node_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_after_add = nn.LayerNorm(hidden_dim) if residual_dt == "add" else None

        for i in range(depth):
            setattr(self, f"gcn_{i}", GCNConv(hidden_dim, hidden_dim, add_self_loops=True))
            setattr(
                self,
                f"kan_{i}",
                KANLinear(hidden_dim, hidden_dim, grid_size=kan_grid_size, bias=True),
            )

        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None

        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        units = [mlp_in] + list(mlp_units) + [1]
        self.mlp_layers = [
            Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)
        ]

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        del edge_features
        h = self.node_proj(node_features)
        dt = self.residual_dt
        for i in range(self.depth):
            msg = getattr(self, f"gcn_{i}")(edge_index, h)
            y = nn.relu(getattr(self, f"kan_{i}")(msg))
            if dt == "add":
                h = nn.relu(self.norm_after_add(h + y))
            elif dt is not None and isinstance(dt, (int, float)):
                h = nn.relu(h + float(dt) * (y - h))
            else:
                h = y
            if training:
                h = self.dropout(h)

        if self.mol_readout is not None:
            graph_repr = self.mol_readout(h, batch_indices, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(h, batch_indices, out_size=num_graphs, aggr="add")

        if self.rdkit_dim > 0 and graph_features is not None:
            graph_repr = mx.concatenate([graph_repr, graph_features], axis=-1)
        if training:
            graph_repr = self.dropout(graph_repr)
        for i, layer in enumerate(self.mlp_layers):
            graph_repr = layer(graph_repr)
            if i < len(self.mlp_layers) - 1:
                graph_repr = nn.leaky_relu(graph_repr)
        return graph_repr


class KAGATRegressor(nn.Module):
    """KA-GAT style regressor for molecular property prediction.
    residual_dt: None = replace; "add" = relu(norm(h+y)); 0.5 = h+0.5*(y-h)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        depth: int = 3,
        dropout: float = 0.1,
        mlp_units: tuple[int, ...] = (64, 32),
        rdkit_dim: int = 0,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        kan_grid_size: int = 8,
        residual_dt: Optional[Union[float, str]] = None,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.depth = depth
        self.residual_dt = residual_dt
        self.rdkit_dim = rdkit_dim
        self.node_proj = Linear(node_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_after_add = nn.LayerNorm(hidden_dim) if residual_dt == "add" else None
        out_per_head = hidden_dim // heads

        for i in range(depth):
            setattr(
                self,
                f"gat_{i}",
                GATConv(
                    hidden_dim,
                    out_per_head,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_features_dim=edge_dim,
                ),
            )
            setattr(
                self,
                f"kan_{i}",
                KANLinear(hidden_dim, hidden_dim, grid_size=kan_grid_size, bias=True),
            )

        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None

        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        units = [mlp_in] + list(mlp_units) + [1]
        self.mlp_layers = [
            Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)
        ]

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features)
        dt = self.residual_dt
        for i in range(self.depth):
            msg = getattr(self, f"gat_{i}")(edge_index, h, edge_features)
            y = nn.relu(getattr(self, f"kan_{i}")(msg))
            if dt == "add":
                h = nn.relu(self.norm_after_add(h + y))
            elif dt is not None and isinstance(dt, (int, float)):
                h = nn.relu(h + float(dt) * (y - h))
            else:
                h = y
            if training:
                h = self.dropout(h)

        if self.mol_readout is not None:
            graph_repr = self.mol_readout(h, batch_indices, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(h, batch_indices, out_size=num_graphs, aggr="add")

        if self.rdkit_dim > 0 and graph_features is not None:
            graph_repr = mx.concatenate([graph_repr, graph_features], axis=-1)
        if training:
            graph_repr = self.dropout(graph_repr)
        for i, layer in enumerate(self.mlp_layers):
            graph_repr = layer(graph_repr)
            if i < len(self.mlp_layers) - 1:
                graph_repr = nn.leaky_relu(graph_repr)
        return graph_repr
