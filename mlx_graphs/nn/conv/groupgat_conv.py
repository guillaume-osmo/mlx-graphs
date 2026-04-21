"""
GroupGAT / AGC: Group-contribution based Graph Attention for molecular property prediction.

Ported from GC-GNN (https://github.com/gsi-lab/GC-GNN).
Reference: Aouichaoui et al., JCIM 2023, https://doi.org/10.1021/acs.jcim.2c01091

Provides:
- GroupGATRegressor: AttentiveFP backbone with group-augmented node features
- GraphODEGroupGATRegressor: ODE-AttFP backbone with group features + optional feature selection
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear
from mlx_graphs.utils import scatter

from mlx_graphs.nn.conv.attentivefp_conv import AttentiveFP
from mlx_graphs.nn.conv.graph_ode_conv import (
    GraphODEBlockAttFP,
    _get_attfp_gru_cell,
    _sample_nodes_for_pooling,
)


class GroupGATRegressor(nn.Module):
    """GroupGAT: AttentiveFP with group-contribution augmented node features.

    Augments each atom with one-hot first-order group membership (Hukkerikar et al.)
    before the AttentiveFP backbone. This injects prior chemical knowledge and
    improves interpretability via attention over group-annotated atoms.

    Input: standard graph (edge_index, node_features, edge_features, batch_indices).
    Node features are expected to be (n_atom,) per atom. The model internally
    concatenates (n_atom + n_groups) and projects to fingerprint_dim.
    """

    def __init__(
        self,
        n_atom: int,
        n_bond: int,
        n_groups: int,
        fingerprint_dim: int = 128,
        radius: int = 2,
        T: int = 2,
        p_dropout: float = 0.1,
        use_rms_norm: bool = False,
        rdkit_dim: int = 0,
        residual_dt: Optional[float | str] = None,
    ):
        super().__init__()
        self.n_atom = n_atom
        self.n_groups = n_groups
        self.n_atom_plus_groups = n_atom + n_groups
        self.attentivefp = AttentiveFP(
            n_atom=self.n_atom_plus_groups,
            n_bond=n_bond,
            fingerprint_dim=fingerprint_dim,
            radius=radius,
            T=T,
            p_dropout=p_dropout,
            use_rms_norm=use_rms_norm,
            rdkit_dim=rdkit_dim,
            residual_dt=residual_dt,
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        group_features: Optional[mx.array] = None,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        """Forward. group_features: (num_nodes, n_groups) one-hot from GroupContributionFeaturizer."""
        if group_features is not None:
            x = mx.concatenate([node_features, group_features], axis=-1)
        else:
            # Pad with zeros if no groups provided
            n = node_features.shape[0]
            pad = mx.zeros((n, self.n_groups), dtype=node_features.dtype)
            x = mx.concatenate([node_features, pad], axis=-1)
        return self.attentivefp(
            edge_index, x, edge_features, batch_indices,
            graph_features=graph_features, training=training,
        )


class GraphODEGroupGATRegressor(nn.Module):
    """ODE-GroupGAT: ODE-AttFP backbone with group-contribution augmented node features.

    Same as GroupGATRegressor but uses ODE integration over AttentiveFP atom-level steps.
    Supports feature selection via masked node/edge inputs (apply mask before forward).
    """

    def __init__(
        self,
        n_atom: int,
        n_bond: int,
        n_groups: int,
        hidden_dim: int = 128,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.1,
        mlp_units: tuple[int, ...] = (64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
        update_mode: str = "ode",
        mol_steps: int = 2,
        mol_integrator: str = "euler",
        learnable_dt: bool = False,
        norm_between_steps: Optional[str] = None,
    ):
        super().__init__()
        self.n_atom = n_atom
        self.n_groups = n_groups
        self.n_atom_plus_groups = n_atom + n_groups
        self.mol_steps = mol_steps
        self.rdkit_dim = rdkit_dim
        self.ode_block = GraphODEBlockAttFP(
            node_dim=self.n_atom_plus_groups,
            edge_dim=n_bond,
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

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        group_features: Optional[mx.array] = None,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        if group_features is not None:
            x = mx.concatenate([node_features, group_features], axis=-1)
        else:
            n = node_features.shape[0]
            pad = mx.zeros((n, self.n_groups), dtype=node_features.dtype)
            x = mx.concatenate([node_features, pad], axis=-1)
        atom_feature = self.ode_block(
            edge_index, x, edge_features, training=training
        )
        atom_pool, batch_pool = _sample_nodes_for_pooling(
            atom_feature, batch_indices, training,
            0, 1.0, False, True,
        )
        num_graphs = int(mx.max(batch_indices).item()) + 1
        if self.mol_steps > 0:
            mol_feature = scatter(
                atom_pool, batch_pool, out_size=num_graphs, aggr="add",
            )
            mol_feature = nn.relu(mol_feature)
            def _mol_context(mol_h: mx.array) -> mx.array:
                mol_expand = mol_h[batch_pool]
                align_inp = mx.concatenate([mol_expand, atom_pool], axis=-1)
                align_score = nn.leaky_relu(self.mol_align(align_inp)).reshape(-1)
                max_per_graph = scatter(
                    align_score, batch_pool, out_size=num_graphs, aggr="max",
                )
                attention_weight = mx.exp(align_score - max_per_graph[batch_pool])
                norm = scatter(
                    attention_weight, batch_pool, out_size=num_graphs, aggr="add",
                )
                attention_weight = attention_weight / (norm[batch_pool] + 1e-8)
                atom_transform = self.mol_attend(
                    self.dropout(atom_pool) if training else atom_pool
                )
                mol_ctx = scatter(
                    attention_weight[:, None] * atom_transform,
                    batch_pool, out_size=num_graphs, aggr="add",
                )
                return nn.elu(mol_ctx)
            def _mol_update(mol_h: mx.array, mol_ctx: mx.array) -> mx.array:
                return nn.relu(self.mol_gru(mol_h, mol_ctx))
            for _ in range(self.mol_steps):
                mol_context = _mol_context(mol_feature)
                mol_feature = _mol_update(mol_feature, mol_context)
            graph_repr = mol_feature
        else:
            graph_repr = scatter(
                atom_pool, batch_pool, out_size=num_graphs, aggr="add",
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
