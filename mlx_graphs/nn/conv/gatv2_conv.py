# GATv2 (Brody et al. ICLR 2022): dynamic attention via score = a^T LeakyReLU(W [h_i || h_j]).
#
# Two critical design pillars (do not skip):
# 1. Edge-to-node projection: project edge features into node (per-head) space before concat,
#    like GINE/GAT, so we never mix raw edge_dim with node dim → avoids dimension explosion.
# 2. One row per edge: flatten to (E, H*att_dim) and att_lin to (E, H*C), not (E*H, ...),
#    so alpha stays (E, H) and scatter index length E matches → no E vs E*H mismatch.

from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.nn.mol_attention_readout import MolAttentionReadout
from mlx_graphs.utils import scatter


class GATv2Conv(MessagePassing):
    """GATv2 convolution: attention score = a^T LeakyReLU(W [h_i || h_j]) (dynamic attention).
    Edge features are projected to node (per-head) space before concat, like GINE/GAT, to avoid
    dimension mismatch; then score = a^T LeakyReLU(W [h_src || h_dst || h_edge])."""

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        heads: int = 1,
        concat: bool = True,
        bias: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_features_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(GATv2Conv, self).__init__(**kwargs)
        self.out_features_dim = out_features_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.lin_proj = Linear(node_features_dim, heads * out_features_dim, bias=False)
        # Pillar 1: edge in same space as node. Pillar 2: att_in/att_out are per-edge (E rows), not per (E,H).
        att_in = 2 * out_features_dim * heads
        if edge_features_dim is not None:
            self.edge_lin_proj = Linear(
                edge_features_dim, heads * out_features_dim, bias=False
            )
            att_in = 3 * out_features_dim * heads
        else:
            self.edge_lin_proj = None
        self.att_lin = Linear(att_in, heads * out_features_dim, bias=False)
        glorot_init = nn.init.glorot_uniform()
        self.att_a = glorot_init(mx.zeros((1, heads, out_features_dim)))
        if bias:
            bias_shape = (heads * out_features_dim) if concat else (out_features_dim)
            self.bias = mx.zeros(bias_shape)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.edge_features_dim = edge_features_dim

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        H, C = self.heads, self.out_features_dim
        N = node_features.shape[0]
        src_idx, dst_idx = edge_index[0], edge_index[1]
        feats = self.lin_proj(node_features).reshape(N, H, C)
        src_feats = feats[src_idx]
        dst_feats = feats[dst_idx]
        E = edge_index.shape[1]
        if self.edge_lin_proj is not None and edge_features is not None:
            if edge_features.ndim == 1:
                edge_features = edge_features.reshape(-1, 1)
            edge_proj = self.edge_lin_proj(edge_features).reshape(E, H, C)
            concat_feats = mx.concatenate([src_feats, dst_feats, edge_proj], axis=-1)
        else:
            concat_feats = mx.concatenate([src_feats, dst_feats], axis=-1)
        # Pillar 2: one row per edge → alpha (E, H) matches dst_idx length E
        concat_flat = concat_feats.reshape(E, -1)
        att_out = self.att_lin(concat_flat).reshape(E, H, C)
        att_out = nn.leaky_relu(att_out, self.negative_slope)
        alpha = (att_out * self.att_a).sum(axis=-1)
        self.num_nodes = N
        # PyG-style: one softmax over edges per (dst, head); scatter_softmax supports (E, H)
        alpha = scatter(alpha, dst_idx, out_size=N, aggr="softmax")
        if "dropout" in self:
            alpha = self.dropout(alpha)
        alpha = alpha.reshape(-1, H, 1)
        msg = alpha * src_feats
        out = scatter(msg, dst_idx, out_size=N, aggr="add")
        if self.concat:
            out = out.reshape(N, H * C)
        else:
            out = mx.mean(out, axis=1)
        if "bias" in self:
            out = out + self.bias
        return out


class GATv2Regressor(nn.Module):
    """GATv2 regressor: stack of GATv2Conv layers + global pool + MLP. Optional rdkit and mol attention.
    residual_dt: None = replace; "add" = relu(norm(h+y)); 0.5 = relu(h+0.5*(y-h))."""

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
        residual_dt: Optional[Union[float, str]] = None,
    ):
        super().__init__()
        assert hidden_dim % heads == 0
        out_per_head = hidden_dim // heads
        self.mol_attention_steps = mol_attention_steps
        self.residual_dt = residual_dt
        self.node_proj = Linear(node_dim, hidden_dim)
        self.depth = depth
        self.norm_after_add = nn.LayerNorm(hidden_dim) if residual_dt == "add" else None
        for i in range(depth):
            setattr(
                self,
                f"gatv2_{i}",
                GATv2Conv(
                    hidden_dim,
                    out_per_head,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_features_dim=edge_dim,
                ),
            )
        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None
        self.dropout = nn.Dropout(dropout)
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        units = [mlp_in] + list(mlp_units) + [1]
        self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
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
        h = self.node_proj(node_features)
        dt = self.residual_dt
        for i in range(self.depth):
            y = getattr(self, f"gatv2_{i}")(edge_index, h, edge_features)
            if dt == "add":
                h = nn.leaky_relu(self.norm_after_add(h + y))
            elif dt is not None and isinstance(dt, (int, float)):
                h = nn.leaky_relu(h + float(dt) * (y - h))
            else:
                h = nn.leaky_relu(y)
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
