# Copyright © 2023-2024 Apple Inc.
# Native MPNN (Gilmer et al.): message = MLP(x_i || x_j || e_ij), aggregate sum, update.
# Reference: Gilmer et al. "Neural Message Passing for Quantum Chemistry" (ICML 2017).

from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.nn.mol_attention_readout import MolAttentionReadout
from mlx_graphs.utils import scatter


class MPNNConv(MessagePassing):
    """Message Passing Neural Network layer (Gilmer et al.).
    Message: MLP(concat(x_src, x_dst, e_ij)); aggregate: sum; update: identity (aggregated)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        out_dim: int,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.message_mlp = nn.Sequential(
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
            raise ValueError("MPNNConv requires edge_features")
        # (E, node_dim), (E, node_dim), (E, edge_dim) -> (E, 2*node_dim + edge_dim)
        inp = mx.concatenate([src_features, dst_features, edge_features], axis=-1)
        return self.message_mlp(inp)

    def update_nodes(self, aggregated: mx.array, **kwargs) -> mx.array:
        return aggregated


class MPNN(nn.Module):
    """Stack of MPNNConv layers with residual and readout for graph-level repr.

    residual_dt: None = classical h = relu(h + m); "add" = norm(h + m) then relu; float 0.5 = relu(h + 0.5*(m - h)).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        depth: int = 5,
        dropout: float = 0.1,
        residual_dt: Optional[Union[float, str]] = None,
    ):
        super().__init__()
        self.residual_dt = residual_dt
        self.node_proj = Linear(node_dim, hidden_dim)
        self.layers = [
            MPNNConv(hidden_dim, edge_dim, hidden_dim)
            for _ in range(depth)
        ]
        self.norm_after_add = nn.LayerNorm(hidden_dim) if residual_dt == "add" else None
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features) if self.node_proj is not None else node_features
        dt = self.residual_dt
        for layer in self.layers:
            m = layer(edge_index, h, edge_features)
            if dt == "add":
                h = nn.relu(self.norm_after_add(h + m))
            elif dt is not None and isinstance(dt, (int, float)):
                h = nn.relu(h + float(dt) * (m - h))
            else:
                h = nn.relu(h + m)
            if training:
                h = self.dropout(h)
        return h


class MPNNRegressor(nn.Module):
    """MPNN regressor for graph-level prediction (e.g. solubility).
    Optional graph_features (e.g. RDKit 217) concatenated before MLP when rdkit_dim > 0.
    mol_attention_steps: if > 0, use AttFP-style mol GRU/attention readout instead of sum pool."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        depth: int = 5,
        dropout: float = 0.1,
        mlp_units: list[int] = (64, 32),
        rdkit_dim: int = 0,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        residual_dt: Optional[Union[float, str]] = None,
    ):
        super().__init__()
        self.mol_attention_steps = mol_attention_steps
        self.mpnn = MPNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            residual_dt=residual_dt,
        )
        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        units = [mlp_in] + list(mlp_units) + [1]
        layers = []
        for i in range(len(units) - 1):
            layers.append(Linear(units[i], units[i + 1], bias=True))
        self.mlp_layers = layers
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        node_repr = self.mpnn(edge_index, node_features, edge_features, training=training)
        if self.mol_readout is not None:
            graph_repr = self.mol_readout(node_repr, batch_indices, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(node_repr, batch_indices, out_size=num_graphs, aggr="add")
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
