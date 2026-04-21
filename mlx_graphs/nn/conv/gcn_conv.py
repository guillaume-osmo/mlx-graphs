from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.nn.mol_attention_readout import MolAttentionReadout
from mlx_graphs.utils import add_self_loops, degree, invert_sqrt_degree
from mlx_graphs.utils import scatter


class GCNConv(MessagePassing):
    """Applies a GCN convolution over input node features.

    Args:
        node_features_dim: size of input node features
        out_features_dim: size of output node embeddings
        bias: whether to use bias in the node projection
        add_self_loops: whether to add a self-loop for each node
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        bias: bool = True,
        add_self_loops: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(GCNConv, self).__init__(**kwargs)

        self.linear = nn.Linear(node_features_dim, out_features_dim, bias)
        self._add_self_loops = add_self_loops

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> mx.array:
        assert edge_index.shape[0] == 2, "edge_index must have shape (2, num_edges)"
        assert (
            edge_index[1].size > 0
        ), "'col' component of edge_index should not be empty"

        node_features = self.linear(node_features)

        if self._add_self_loops:
            edge_index = add_self_loops(edge_index)

        row, col = edge_index

        # Compute node degree normalization for the mean aggregation.
        norm: Optional[mx.array] = None
        if normalize:
            deg = degree(col, node_features.shape[0], edge_weights=edge_weights)
            # NOTE : need boolean indexing in order to zero out inf values
            deg_inv_sqrt = invert_sqrt_degree(deg)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Compute messages and aggregate them with sum and norm.
        node_features = self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_weights": norm},
        )

        return node_features


class GCNRegressor(nn.Module):
    """GCN regressor: stack of GCNConv + global pool + MLP.
    residual_dt: None = replace h=relu(conv(h)); "add" = relu(norm(h+conv(h))); 0.5 = relu(h+0.5*(conv(h)-h)).
    """

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
        residual_dt: Optional[Union[float, str]] = None,
    ):
        del edge_dim  # GCN variant does not use edge features directly.
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
        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        units = [mlp_in] + list(mlp_units) + [1]
        self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]

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
            y = getattr(self, f"gcn_{i}")(edge_index, h)
            if dt == "add":
                h = nn.relu(self.norm_after_add(h + y))
            elif dt is not None and isinstance(dt, (int, float)):
                h = nn.relu(h + float(dt) * (y - h))
            else:
                h = nn.relu(y)
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
