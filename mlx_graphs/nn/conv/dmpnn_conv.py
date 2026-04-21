# Copyright © 2023-2024 Apple Inc.
# DMPNN: Directed Message Passing Neural Network (ChemProp-style).
# Uses directed graph: each edge (i->j) has a reverse edge (j->i); messages
# are passed along directed edges and aggregated excluding the reverse edge.
# Reference: Yang et al. (2019) "Analyzing Learned Molecular Representations
# for Property Prediction" https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237
# KGCNN reference: https://github.com/osmoai/kgcnn-keras-unlocked (DMPNN).

from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.mol_attention_readout import MolAttentionReadout
from mlx_graphs.utils import scatter


def to_directed_edge_index_and_reverse(
    edge_index: mx.array,
) -> tuple[mx.array, mx.array]:
    """Build directed edge_index and reverse indices from undirected edge_index.

    Each undirected edge (a, b) becomes two directed edges (a->b) and (b->a).
    edge_reverse[e] is the index of the directed edge that is the reverse of e.

    Args:
        edge_index: Undirected [2, num_edges], edge_index[0]=src, edge_index[1]=dst.

    Returns:
        directed_edge_index: [2, num_directed_edges] with both (a,b) and (b,a).
        edge_reverse: [num_directed_edges] int32, edge_reverse[e] = index of reverse edge.
    """
    src, dst = edge_index[0], edge_index[1]
    num_undir = src.shape[0]
    # Directed: first half (a->b), second half (b->a)
    dir_src = mx.concatenate([src, dst], axis=0)
    dir_dst = mx.concatenate([dst, src], axis=0)
    directed_edge_index = mx.stack([dir_src, dir_dst], axis=0)
    # Reverse: edge at i has reverse at i + num_undir (or i - num_undir)
    edge_reverse = mx.concatenate([
        mx.arange(num_undir, 2 * num_undir, dtype=mx.int32),
        mx.arange(num_undir, dtype=mx.int32),
    ], axis=0)
    return directed_edge_index, edge_reverse


def edge_reverse_from_directed_pairs(num_edges: int) -> mx.array:
    """Build edge_reverse when directed edges are stored in consecutive pairs.

    If edge_index is built so that for each bond (a,b), edges are (a->b) at index 2k
    and (b->a) at index 2k+1, then edge_reverse[i] = i+1 for even i and i-1 for odd i.

    Args:
        num_edges: Total number of directed edges (must be even).

    Returns:
        edge_reverse: [num_edges] int32, edge_reverse[e] = index of reverse edge.
    """
    import numpy as np
    rev = np.empty(num_edges, dtype=np.int32)
    rev[0::2] = np.arange(1, num_edges, 2)
    rev[1::2] = np.arange(0, num_edges, 2)
    return mx.array(rev)


class DMPNNConv(nn.Module):
    """One DMPNN message-passing step on directed edges.

    For each directed edge (i->j): new message = sum(messages into i) - message(j->i),
    then h_new = activation(W @ new_message + h0). Requires edge_reverse so the
    reverse edge can be excluded from the sum. edge_hidden_dim must equal
    edge_dense_units for the skip connection (h0) to match.
    """

    def __init__(
        self,
        edge_hidden_dim: int,
        edge_initialize_units: Optional[int] = None,
        edge_dense_units: Optional[int] = None,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        ed = edge_dense_units if edge_dense_units is not None else edge_hidden_dim
        self.edge_dense = Linear(edge_hidden_dim, ed, bias=True)
        if activation == "relu":
            self.activation = nn.relu
        elif activation == "leaky_relu":
            self.activation = nn.leaky_relu
        else:
            self.activation = nn.leaky_relu

    def __call__(
        self,
        directed_edge_index: mx.array,
        edge_reverse: mx.array,
        edge_h: mx.array,
        h0: mx.array,
    ) -> mx.array:
        """One DMPNN step: aggregate into source nodes, subtract reverse, update.

        Args:
            directed_edge_index: [2, num_directed_edges], row0=src, row1=dst.
            edge_reverse: [num_directed_edges], edge_reverse[e] = index of reverse edge.
            edge_h: [num_directed_edges, edge_hidden_dim] current edge messages.
            h0: [num_directed_edges, edge_hidden_dim] initial edge embeddings (skip).

        Returns:
            New edge messages [num_directed_edges, edge_dense_units] (after dense + skip + act).
        """
        num_nodes = int(mx.max(directed_edge_index).item()) + 1
        dst_idx = directed_edge_index[1]
        src_idx = directed_edge_index[0]
        # 1) Sum edge messages by destination node -> per-node sum of incoming messages
        pool_to_nodes = scatter(
            edge_h, dst_idx, out_size=num_nodes, aggr="add"
        )
        # 2) For each edge (i->j), take value at source i (sum of messages into i)
        msg_into_src = pool_to_nodes[src_idx]
        # 3) Subtract the reverse edge's message (exclude j->i from sum at i)
        reverse_msg = edge_h[edge_reverse]
        m_vw = msg_into_src - reverse_msg
        # 4) Update: h_new = act(W @ m_vw + h0)
        h_new = self.edge_dense(m_vw) + h0
        return self.activation(h_new)


class DMPNN(nn.Module):
    """DMPNN backbone: directed message passing then node aggregation.

    Expects directed edge_index and edge_reverse (use to_directed_edge_index_and_reverse
    if you have undirected edges). Node and edge features are projected to hidden_dim;
    initial edge state h0 = act(W0 @ concat(node_src, edge_feat)); then depth steps
    of DMPNNConv; then node repr = concat(sum(edge_h by dest), node_feat) -> dense.

    residual_dt: Update rule at each layer (all use same stack: depth layers, no weight sharing).
      None: original DMPNN — h = layer(h, h0). Skip is only h0 inside the layer (h_new = act(W@m_vw + h0)).
      float 0.5: SCORE-style — h = h + 0.5 * (layer(h, h0) - h).
      "add": classical residual (ResNet-style) — h = h + layer(h, h0).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        depth: int = 5,
        edge_initialize_units: Optional[int] = None,
        edge_dense_units: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "leaky_relu",
        residual_dt: Optional[Union[float, str]] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.residual_dt = residual_dt
        self.activation_name = activation
        act_fn = nn.leaky_relu if activation == "leaky_relu" else nn.relu
        self.act = act_fn
        eu = edge_initialize_units or hidden_dim
        ed = edge_dense_units or hidden_dim
        self.edge_init = Linear(node_dim + edge_dim, eu, bias=True)
        # Skip connection uses h0, so conv output dim must match eu
        self.conv_layers = [
            DMPNNConv(eu, edge_dense_units=eu, activation=activation) for _ in range(depth)
        ]
        # Normalize after classical residual to prevent h from blowing up (NaN).
        self.norm_after_add = nn.LayerNorm(eu) if residual_dt == "add" else None
        self.node_dense = Linear(eu + node_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        directed_edge_index: mx.array,
        edge_reverse: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        """Forward: initial edge h0, depth steps of DMPNN, then node aggregation.

        Returns:
            Node embeddings [num_nodes, hidden_dim].
        """
        src_idx = directed_edge_index[0]
        dst_idx = directed_edge_index[1]
        num_nodes = node_features.shape[0]
        # Initial edge hidden: concat(source node feat, edge feat) -> dense -> act
        node_src = node_features[src_idx]
        inp = mx.concatenate([node_src, edge_features], axis=-1)
        h0 = self.act(self.edge_init(inp))
        h = h0
        dt = self.residual_dt
        for layer in self.conv_layers:
            y = layer(directed_edge_index, edge_reverse, h, h0)
            if dt == "add":
                h = self.norm_after_add(h + y)  # classical residual + norm to avoid explosion
            elif dt is not None and isinstance(dt, (int, float)):
                h = h + float(dt) * (y - h)  # SCORE-style blend
            else:
                h = y  # original: replace
            if training:
                h = self.dropout(h)
        # Aggregate to nodes: sum edge messages by destination, concat node feat
        node_from_edges = scatter(h, dst_idx, out_size=num_nodes, aggr="add")
        node_repr = mx.concatenate([node_from_edges, node_features], axis=-1)
        return self.act(self.node_dense(node_repr))  # [num_nodes, hidden_dim]


class DMPNNRegressor(nn.Module):
    """DMPNN regressor for molecular property prediction (graph-level readout).

    Uses directed graph: build directed_edge_index and edge_reverse from
    undirected edge_index via to_directed_edge_index_and_reverse().

    Optional graph-level features (e.g. RDKit 217 descriptors) can be
    concatenated with the graph embedding before the MLP via graph_features
    when rdkit_dim > 0.

    mol_attention_steps: if > 0, use AttFP-style mol GRU/attention readout instead of sum pool.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        depth: int = 5,
        dropout: float = 0.1,
        mlp_units: list[int] = (128, 64, 32),
        rdkit_dim: int = 0,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        mlp_ode_steps: int = 0,
        mlp_ode_dt: float = 0.25,
        mlp_ode_integrator: str = "euler",
        residual_dt: Optional[Union[float, str]] = None,
    ):
        super().__init__()
        self.mol_attention_steps = mol_attention_steps
        self.mlp_ode_steps = mlp_ode_steps
        self.dmpnn = DMPNN(
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
        if mlp_ode_steps > 0:
            from mlx_graphs.nn.conv.graph_ode_conv import MLPODEBlock
            first_dim = mlp_units[0] if mlp_units else 128
            self.mlp_ode = MLPODEBlock(
                mlp_in, first_dim, mlp_ode_steps, mlp_ode_dt, mlp_ode_integrator
            )
            self.mlp_layers = []
        else:
            self.mlp_ode = None
            units = [mlp_in] + list(mlp_units) + [1]
            self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.rdkit_norm = nn.LayerNorm(rdkit_dim) if rdkit_dim > 0 else None

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
        """Forward; batch_indices maps each node to its graph index (0 to num_graphs-1).
        If rdkit_dim > 0, graph_features [num_graphs, rdkit_dim] are concatenated
        with the graph embedding before the MLP."""
        node_repr = self.dmpnn(
            directed_edge_index,
            edge_reverse,
            node_features,
            edge_features,
            training=training,
        )
        if self.mol_readout is not None:
            graph_repr = self.mol_readout(node_repr, batch_indices, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(
                node_repr, batch_indices, out_size=num_graphs, aggr="add"
            )
        if self.rdkit_dim > 0 and graph_features is not None and self.rdkit_norm is not None:
            h = mx.concatenate([graph_repr, self.rdkit_norm(graph_features)], axis=-1)
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
        return h  # (num_graphs, 1) to match AttentiveFP / mse_loss targets


class KADMPNN(DMPNN):
    """KADMPNN backbone: DMPNN with KAN transform on edge/node updates."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        depth: int = 5,
        edge_initialize_units: Optional[int] = None,
        edge_dense_units: Optional[int] = None,
        dropout: float = 0.1,
        kan_grid_size: int = 8,
    ):
        super().__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            edge_initialize_units=edge_initialize_units,
            edge_dense_units=edge_dense_units,
            dropout=dropout,
        )
        from mlx_graphs.nn.conv.ka_gnn_conv import KANLinear
        eu = edge_initialize_units or hidden_dim
        self.edge_kan = KANLinear(eu, eu, grid_size=kan_grid_size, bias=True)
        self.node_kan = KANLinear(hidden_dim, hidden_dim, grid_size=kan_grid_size, bias=True)
        self.edge_ka_residual_scale = mx.array([0.1], dtype=mx.float32)
        self.node_ka_residual_scale = mx.array([0.1], dtype=mx.float32)

    def __call__(
        self,
        directed_edge_index: mx.array,
        edge_reverse: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        src_idx = directed_edge_index[0]
        dst_idx = directed_edge_index[1]
        num_nodes = node_features.shape[0]
        node_src = node_features[src_idx]
        inp = mx.concatenate([node_src, edge_features], axis=-1)
        h0 = nn.relu(self.edge_init(inp))
        h = h0
        for layer in self.conv_layers:
            h = layer(directed_edge_index, edge_reverse, h, h0)
            h_safe = 3.0 * mx.tanh(h / 3.0)
            h = nn.relu(h + self.edge_ka_residual_scale[0] * self.edge_kan(h_safe))
            if training:
                h = self.dropout(h)
        node_from_edges = scatter(h, dst_idx, out_size=num_nodes, aggr="add")
        node_repr = mx.concatenate([node_from_edges, node_features], axis=-1)
        node_repr = nn.relu(self.node_dense(node_repr))
        node_safe = 3.0 * mx.tanh(node_repr / 3.0)
        return nn.relu(node_repr + self.node_ka_residual_scale[0] * self.node_kan(node_safe))


class KADMPNNRegressor(nn.Module):
    """KADMPNN regressor for molecular property prediction."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        depth: int = 5,
        dropout: float = 0.1,
        mlp_units: list[int] = (128, 64, 32),
        rdkit_dim: int = 0,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        mlp_ode_steps: int = 0,
        mlp_ode_dt: float = 0.25,
        mlp_ode_integrator: str = "euler",
        kan_grid_size: int = 8,
    ):
        super().__init__()
        self.mlp_ode_steps = mlp_ode_steps
        self.kadmpnn = KADMPNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
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
            from mlx_graphs.nn.conv.graph_ode_conv import MLPODEBlock
            first_dim = mlp_units[0] if mlp_units else 128
            self.mlp_ode = MLPODEBlock(
                mlp_in, first_dim, mlp_ode_steps, mlp_ode_dt, mlp_ode_integrator
            )
            self.mlp_layers = []
        else:
            self.mlp_ode = None
            units = [mlp_in] + list(mlp_units) + [1]
            self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_dim = rdkit_dim
        self.rdkit_norm = nn.LayerNorm(rdkit_dim) if rdkit_dim > 0 else None

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
        node_repr = self.kadmpnn(
            directed_edge_index,
            edge_reverse,
            node_features,
            edge_features,
            training=training,
        )
        if self.mol_readout is not None:
            graph_repr = self.mol_readout(node_repr, batch_indices, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(node_repr, batch_indices, out_size=num_graphs, aggr="add")
        if self.rdkit_dim > 0 and graph_features is not None and self.rdkit_norm is not None:
            h = mx.concatenate([graph_repr, self.rdkit_norm(graph_features)], axis=-1)
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
