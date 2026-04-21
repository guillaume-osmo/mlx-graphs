# Copyright © 2023-2024 Apple Inc.
# Graph Transformer: multi-head self-attention over nodes with graph-level masking
# (nodes only attend to nodes in the same graph).
# Optional edge features: additive attention bias (per head) and additive value (gt-pyg style).

from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear
from mlx_graphs.utils import scatter
from mlx_graphs.utils.scatter import scatter_add


class GraphTransformerBlock(nn.Module):
    """Single block: multi-head self-attention (masked by batch) + FFN + residual.
    When edge_dim > 0 and use_edge_features: edge features add bias to attention logits
    and an additive contribution to values (gt-pyg style).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        edge_dim: int = 0,
        dropout: float = 0.1,
        ffn_dim: Optional[int] = None,
        use_edge_features: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim
        self.scale = self.head_dim ** (-0.5)
        self.edge_dim = edge_dim
        self.use_edge_features = use_edge_features and (edge_dim > 0)
        self.q_proj = Linear(hidden_dim, hidden_dim)
        self.k_proj = Linear(hidden_dim, hidden_dim)
        self.v_proj = Linear(hidden_dim, hidden_dim)
        self.out_proj = Linear(hidden_dim, hidden_dim)
        if self.use_edge_features:
            self.we_logits = Linear(edge_dim, num_heads, bias=True)   # edge -> attention bias per head
            self.we_value = Linear(edge_dim, hidden_dim, bias=True)    # edge -> value contribution
        else:
            self.we_logits = None
            self.we_value = None
        self.ffn = nn.Sequential(
            Linear(hidden_dim, ffn_dim or hidden_dim * 4),
            nn.ReLU(),
            Linear(ffn_dim or hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        node_features: mx.array,
        batch_indices: mx.array,
        edge_index: Optional[mx.array] = None,
        edge_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        N = node_features.shape[0]
        # (N, H, D); we need scores (N, H, N) so that scores[i,h,j] = q[i,h,:]·k[j,h,:]
        q = self._reshape_heads(self.q_proj(node_features))  # (N, H, D)
        k = self._reshape_heads(self.k_proj(node_features))  # (N, H, D)
        v = self._reshape_heads(self.v_proj(node_features))  # (N, H, D)
        # (H, N, D) @ (H, D, N) -> (H, N, N), then (N, H, N)
        q_t = mx.transpose(q, (1, 0, 2))   # (H, N, D)
        k_t = mx.transpose(k, (1, 0, 2))   # (H, N, D)
        scores_t = mx.matmul(q_t, mx.swapaxes(k_t, -2, -1)) * self.scale  # (H, N, N)
        scores = mx.transpose(scores_t, (1, 0, 2))  # (N, H, N)

        # Mask: only allow attention within same graph
        batch_i = batch_indices.reshape(-1, 1, 1)
        batch_j = batch_indices.reshape(1, 1, -1)
        mask = mx.where(batch_i == batch_j, 0.0, -1e9)
        scores = scores + mask

        # Edge bias: add to scores at (dst, src) for each edge
        if self.use_edge_features and edge_index is not None and edge_features is not None:
            E = edge_index.shape[1]
            if edge_features.ndim == 1:
                edge_features = edge_features.reshape(-1, 1)
            edge_bias = self.we_logits(edge_features)  # (E, num_heads)
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            linear_idx = dst_idx * N + src_idx  # (E,)
            bias_flat = mx.zeros((N * N, self.num_heads), dtype=scores.dtype)
            bias_flat = scatter_add(bias_flat, linear_idx, edge_bias)
            bias_mat = mx.reshape(bias_flat, (N, N, self.num_heads))  # (N, N, H)
            scores = scores + mx.transpose(bias_mat, (0, 2, 1))  # (N, H, N)

        attn = mx.softmax(scores, axis=-1)
        if training:
            attn = self.dropout(attn)
        # (H, N, N) @ (H, N, D) -> (H, N, D), then (N, H, D)
        attn_t = mx.transpose(attn, (1, 0, 2))   # (H, N, N)
        v_t = mx.transpose(v, (1, 0, 2))         # (H, N, D)
        out_t = mx.matmul(attn_t, v_t)           # (H, N, D)
        out = mx.transpose(out_t, (1, 0, 2))    # (N, H, D)

        # Edge value: additive contribution attn[i,h,j] * edge_val[i,j,h,:]
        if self.use_edge_features and edge_index is not None and edge_features is not None:
            edge_val = self.we_value(edge_features)  # (E, hidden_dim)
            edge_val = mx.reshape(edge_val, (E, self.num_heads, self.head_dim))  # (E, H, D)
            val_flat = mx.zeros((N * N, self.num_heads, self.head_dim), dtype=v.dtype)
            val_flat = scatter_add(val_flat, linear_idx, edge_val)
            edge_val_mat = mx.reshape(val_flat, (N, N, self.num_heads, self.head_dim))  # (N, N, H, D)
            edge_val_mat = mx.transpose(edge_val_mat, (0, 2, 1, 3))  # (N, H, N, D)
            contrib = mx.sum(attn[:, :, :, None] * edge_val_mat, axis=2)  # (N, H, D)
            out = out + contrib

        out = self._flatten_heads(out)
        out = self.out_proj(out)
        out = self.dropout(out) if training else out
        h = self.norm1(node_features + out)
        h = h + (self.dropout(self.ffn(h)) if training else self.ffn(h))
        return self.norm2(h)

    def _reshape_heads(self, x: mx.array) -> mx.array:
        n, d = x.shape
        return mx.reshape(x, (n, self.num_heads, self.head_dim))

    def _flatten_heads(self, x: mx.array) -> mx.array:
        n, h, d = x.shape
        return mx.reshape(x, (n, h * d))


class GraphTransformer(nn.Module):
    """Stack of GraphTransformer blocks.
    residual_dt: None = replace h=block(h); "add" = norm(h+block(h)); 0.5 = h+0.5*(block(h)-h).
    use_edge_features: when True and edge_dim > 0, edges add attention bias and value contribution (gt-pyg style)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        depth: int = 3,
        dropout: float = 0.1,
        ffn_dim: Optional[int] = None,
        residual_dt: Optional[Union[float, str]] = None,
        use_edge_features: bool = True,
    ):
        super().__init__()
        self.residual_dt = residual_dt
        self.node_proj = Linear(node_dim, hidden_dim)
        self.norm_after_add = nn.LayerNorm(hidden_dim) if residual_dt == "add" else None
        self.blocks = [
            GraphTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                edge_dim=edge_dim,
                dropout=dropout,
                ffn_dim=ffn_dim,
                use_edge_features=use_edge_features,
            )
            for _ in range(depth)
        ]
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        node_features: mx.array,
        batch_indices: mx.array,
        edge_index: Optional[mx.array] = None,
        edge_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        h = self.node_proj(node_features)
        dt = self.residual_dt
        for block in self.blocks:
            y = block(h, batch_indices, edge_index, edge_features, training=training)
            if dt == "add":
                h = self.norm_after_add(h + y)
            elif dt is not None and isinstance(dt, (int, float)):
                h = h + float(dt) * (y - h)
            else:
                h = y
        if training:
            h = self.dropout(h)
        return h


class GraphTransformerRegressor(nn.Module):
    """Graph Transformer regressor: transformer stack + global pool + MLP.
    Optional graph_features (e.g. RDKit 217) when rdkit_dim > 0.
    residual_dt: None = replace; "add" = norm(h+block(h)); 0.5 = h+0.5*(block(h)-h).
    use_edge_features: when True and edge_dim > 0, edges add attention bias + value (gt-pyg style)."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        depth: int = 3,
        dropout: float = 0.1,
        mlp_units: list[int] = (64, 32),
        rdkit_dim: int = 0,
        residual_dt: Optional[Union[float, str]] = None,
        use_edge_features: bool = True,
    ):
        super().__init__()
        self.transformer = GraphTransformer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            depth=depth,
            dropout=dropout,
            residual_dt=residual_dt,
            use_edge_features=use_edge_features,
        )
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
        node_repr = self.transformer(
            node_features, batch_indices, edge_index, edge_features, training=training
        )
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
