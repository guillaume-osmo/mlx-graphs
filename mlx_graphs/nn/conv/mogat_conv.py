# Copyright © 2023-2024 Apple Inc.
# MoGAT: Multi-order Graph Attention Network for molecular property prediction.
# Lee et al., Scientific Reports (2023) 13:957, https://doi.org/10.1038/s41598-022-25701-5
#
# Key idea: Extract graph embeddings at EVERY node embedding layer (multi-order neighbors),
# then merge them by attention to form the final graph embedding. This improves prediction
# and interpretability compared to AttentiveFP which only uses the last layer.
#
# ODE-MoGAT: Same logic but with ODE integration. Graph embeddings at EACH ODE step
# are aggregated via attention (concat/merge over the ODE trajectory).

from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear
from mlx_graphs.utils import scatter


def _ode_step_euler(
    h: mx.array, dt: float, step_fn: Callable[[mx.array], mx.array]
) -> mx.array:
    return h + dt * (step_fn(h) - h)


def _get_gru_cell():
    """Lazy import of AttentiveFP GRU cell to avoid circular import."""
    from mlx_graphs.nn.conv.attentivefp_conv import _GRUCell
    return _GRUCell


class MoGATRegressor(nn.Module):
    """MoGAT: Multi-order Graph Attention Network for molecular property prediction.

    Unlike AttentiveFP which only uses the final layer's graph embedding, MoGAT:
    1. Computes a graph embedding after EACH node embedding layer (different neighboring orders)
    2. Merges graph embeddings via attention: final = softmax(scores) @ stacked_embeddings
    3. Predicts from the final graph embedding

    Paper: Lee et al., Sci. Rep. (2023) 13:957.
    Dataset: ESOL (water solubility), RMSE 0.4784.
    """

    def __init__(
        self,
        n_atom: int,
        n_bond: int,
        fingerprint_dim: int = 200,
        radius: int = 3,
        mol_steps: int = 2,
        mol_attention_steps: int = 0,
        p_dropout: float = 0.2,
        rdkit_dim: int = 0,
    ):
        super().__init__()
        self.radius = radius
        self.mol_steps = mol_steps
        self.mol_attention_steps = mol_attention_steps
        self.fingerprint_dim = fingerprint_dim
        self.rdkit_dim = rdkit_dim

        _GRUCell = _get_gru_cell()
        self.atom_fc = Linear(n_atom, fingerprint_dim)
        self.neighbor_fc = Linear(n_atom + n_bond, fingerprint_dim)
        self.gru_layers = [_GRUCell(fingerprint_dim, fingerprint_dim) for _ in range(radius)]
        self.align_layers = [Linear(2 * fingerprint_dim, 1) for _ in range(radius)]
        self.attend_layers = [Linear(fingerprint_dim, fingerprint_dim) for _ in range(radius)]

        # Super-node (graph embedding) for each layer: attends to all atoms, GRU update
        self.super_align = Linear(2 * fingerprint_dim, 1)
        self.super_attend = Linear(fingerprint_dim, fingerprint_dim)
        self.super_gru = _GRUCell(fingerprint_dim, fingerprint_dim)

        # AttentiveFP-style mol-level attention (optional, for fair comparison with AttFP)
        if mol_attention_steps > 0:
            self.mol_align = Linear(2 * fingerprint_dim, 1)
            self.mol_attend = Linear(fingerprint_dim, fingerprint_dim)
            self.mol_gru = _GRUCell(fingerprint_dim, fingerprint_dim)
        else:
            self.mol_align = None
            self.mol_attend = None
            self.mol_gru = None

        # Attention over graph embeddings from all layers (Eq. 8 style)
        self.layer_attention = Linear(fingerprint_dim, 1)

        mlp_in = fingerprint_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        self.mlp_fc1 = Linear(mlp_in, 64)
        self.mlp_fc2 = Linear(64, 32)
        self.mlp_out = Linear(32, 1)
        self.dropout = nn.Dropout(p_dropout)
        if rdkit_dim > 0:
            self.rdkit_norm = nn.LayerNorm(rdkit_dim)
        else:
            self.rdkit_norm = None

    def _node_step(
        self,
        edge_index: mx.array,
        h: mx.array,
        neighbor_feat: mx.array,
        align: Linear,
        attend: Linear,
        gru_cell: nn.Module,
        training: bool,
    ) -> mx.array:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        num_nodes = h.shape[0]
        dst_feat = h[dst_idx]
        align_inp = mx.concatenate([dst_feat, neighbor_feat], axis=-1)
        align_score = nn.leaky_relu(align(align_inp))
        align_score = align_score.reshape(-1)
        max_per_dst = scatter(align_score, dst_idx, out_size=num_nodes, aggr="max")
        attn_w = mx.exp(align_score - max_per_dst[dst_idx])
        norm = scatter(attn_w, dst_idx, out_size=num_nodes, aggr="add")
        attn_w = attn_w / (norm[dst_idx] + 1e-8)
        neighbor_transform = attend(
            self.dropout(neighbor_feat) if training else neighbor_feat
        )
        context = scatter(
            attn_w[:, None] * neighbor_transform,
            dst_idx,
            out_size=num_nodes,
            aggr="add",
        )
        context = nn.elu(context)
        h_new = gru_cell(h, context)
        return nn.leaky_relu(h_new)

    def _graph_embedding_per_layer(
        self,
        atom_feature: mx.array,
        batch_indices: mx.array,
        num_graphs: int,
        training: bool,
    ) -> mx.array:
        """Compute graph embedding via virtual super node attending to all atoms.
        Batched with scatter (no GPU sync, no per-graph Python loop).
        Returns (num_graphs, fingerprint_dim)."""
        h_super = mx.zeros((num_graphs, self.fingerprint_dim))
        for _ in range(self.mol_steps):
            mol_expand = h_super[batch_indices]
            align_inp = mx.concatenate([mol_expand, atom_feature], axis=-1)
            align_score = nn.leaky_relu(self.super_align(align_inp)).reshape(-1)
            max_per_graph = scatter(
                align_score, batch_indices, out_size=num_graphs, aggr="max"
            )
            attn_w = mx.exp(align_score - max_per_graph[batch_indices])
            norm = scatter(attn_w, batch_indices, out_size=num_graphs, aggr="add")
            attn_w = attn_w / (norm[batch_indices] + 1e-8)
            neighbor_transform = self.super_attend(
                self.dropout(atom_feature) if training else atom_feature
            )
            context = scatter(
                attn_w[:, None] * neighbor_transform,
                batch_indices,
                out_size=num_graphs,
                aggr="add",
            )
            context = nn.elu(context)
            h_super = self.super_gru(h_super, context)
            h_super = nn.leaky_relu(h_super)
        return h_super

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        num_nodes = node_features.shape[0]
        num_graphs = int(mx.max(batch_indices).item()) + 1

        src_idx = edge_index[0]
        neighbor_feat = mx.concatenate(
            [node_features[src_idx], edge_features], axis=-1
        )
        neighbor_feat = nn.leaky_relu(self.neighbor_fc(neighbor_feat))
        atom_feature = nn.leaky_relu(self.atom_fc(node_features))

        graph_embeddings_list = []
        for d in range(self.radius):
            atom_feature = self._node_step(
                edge_index,
                atom_feature,
                neighbor_feat,
                self.align_layers[d],
                self.attend_layers[d],
                self.gru_layers[d],
                training,
            )
            g_d = self._graph_embedding_per_layer(
                atom_feature, batch_indices, num_graphs, training
            )
            graph_embeddings_list.append(g_d)

        # Stack: (num_graphs, radius, d) then attention over radius
        G_stack = mx.stack(graph_embeddings_list, axis=1)
        scores = self.layer_attention(G_stack).squeeze(-1)
        attn_weights = mx.softmax(scores, axis=-1)
        mol_feature = mx.sum(
            attn_weights[:, :, None] * G_stack, axis=1
        )

        # AttentiveFP-style mol-level refinement (for fair comparison)
        if self.mol_attention_steps > 0 and self.mol_align is not None:
            for _ in range(self.mol_attention_steps):
                mol_expand = mol_feature[batch_indices]
                align_inp = mx.concatenate([mol_expand, atom_feature], axis=-1)
                align_score = nn.leaky_relu(self.mol_align(align_inp)).reshape(-1)
                max_per_graph = scatter(
                    align_score, batch_indices, out_size=num_graphs, aggr="max"
                )
                attention_weight = mx.exp(
                    align_score - max_per_graph[batch_indices]
                )
                norm = scatter(
                    attention_weight, batch_indices, out_size=num_graphs, aggr="add"
                )
                attention_weight = attention_weight / (norm[batch_indices] + 1e-8)
                atom_transform = self.mol_attend(
                    self.dropout(atom_feature) if training else atom_feature
                )
                mol_context = scatter(
                    attention_weight[:, None] * atom_transform,
                    batch_indices,
                    out_size=num_graphs,
                    aggr="add",
                )
                mol_context = nn.elu(mol_context)
                mol_feature = self.mol_gru(mol_feature, mol_context)
                mol_feature = nn.leaky_relu(mol_feature)

        if self.rdkit_dim > 0 and graph_features is not None and self.rdkit_norm is not None:
            graph_features_norm = self.rdkit_norm(graph_features)
            r0 = mx.concatenate([mol_feature, graph_features_norm], axis=-1)
        else:
            r0 = mol_feature
        r0 = self.dropout(r0) if training else r0
        h = nn.leaky_relu(self.mlp_fc1(r0))
        h = nn.leaky_relu(self.mlp_fc2(h))
        return self.mlp_out(h)


class GraphODEMoGATRegressor(nn.Module):
    """ODE-MoGAT: ODE integration + multi-order graph embedding aggregation.

    Same logic as MoGAT but with ODE dynamics:
    - Node updates: ODE integration (AttFP-style step as velocity)
    - At EACH ODE step: compute graph embedding via super-node attention
    - Merge graph embeddings from all steps via attention → final prediction

    Analogy: MoGAT uses graph embeddings from each message-passing layer;
    ODE-MoGAT uses graph embeddings from each ODE integration step.
    """

    def __init__(
        self,
        n_atom: int,
        n_bond: int,
        hidden_dim: int = 128,
        ode_steps: int = 4,
        ode_dt: float = 0.25,
        dropout: float = 0.2,
        mol_steps: int = 2,
        mol_attention_steps: int = 0,
        mlp_units: tuple[int, ...] = (64, 32),
        rdkit_dim: int = 0,
        integrator: str = "euler",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ode_steps = ode_steps
        self.ode_dt = ode_dt
        self.integrator = integrator
        self.mol_steps = mol_steps
        self.mol_attention_steps = mol_attention_steps
        self.rdkit_dim = rdkit_dim

        from mlx_graphs.nn.conv.graph_ode_conv import (
            _ode_integrate,
            _get_attfp_node_step,
        )

        self._ode_integrate_fn = _ode_integrate
        AttentiveFPNodeStep = _get_attfp_node_step()
        self.node_proj = Linear(n_atom, hidden_dim)
        self.attfp_step = AttentiveFPNodeStep(
            fp_dim=hidden_dim, edge_dim=n_bond, dropout=dropout
        )

        _GRUCell = _get_gru_cell()
        self.super_align = Linear(2 * hidden_dim, 1)
        self.super_attend = Linear(hidden_dim, hidden_dim)
        self.super_gru = _GRUCell(hidden_dim, hidden_dim)

        # AttentiveFP-style mol-level attention (optional, for fair comparison)
        if mol_attention_steps > 0:
            self.mol_align = Linear(2 * hidden_dim, 1)
            self.mol_attend = Linear(hidden_dim, hidden_dim)
            self.mol_gru = _GRUCell(hidden_dim, hidden_dim)
        else:
            self.mol_align = None
            self.mol_attend = None
            self.mol_gru = None

        self.layer_attention = Linear(hidden_dim, 1)
        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        units = [mlp_in] + list(mlp_units) + [1]
        self.mlp_layers = [
            Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)
        ]
        self.dropout = nn.Dropout(dropout)
        self.rdkit_norm = nn.LayerNorm(rdkit_dim) if rdkit_dim > 0 else None

    def _graph_embedding_from_atoms(
        self,
        atom_feature: mx.array,
        batch_indices: mx.array,
        num_graphs: int,
        training: bool,
    ) -> mx.array:
        """Batched with scatter (no GPU sync, no per-graph Python loop)."""
        h_super = mx.zeros((num_graphs, self.hidden_dim))
        for _ in range(self.mol_steps):
            mol_expand = h_super[batch_indices]
            align_inp = mx.concatenate([mol_expand, atom_feature], axis=-1)
            align_score = nn.leaky_relu(self.super_align(align_inp)).reshape(-1)
            max_per_graph = scatter(
                align_score, batch_indices, out_size=num_graphs, aggr="max"
            )
            attn_w = mx.exp(align_score - max_per_graph[batch_indices])
            norm = scatter(attn_w, batch_indices, out_size=num_graphs, aggr="add")
            attn_w = attn_w / (norm[batch_indices] + 1e-8)
            neighbor_transform = self.super_attend(
                self.dropout(atom_feature) if training else atom_feature
            )
            context = scatter(
                attn_w[:, None] * neighbor_transform,
                batch_indices,
                out_size=num_graphs,
                aggr="add",
            )
            context = nn.elu(context)
            h_super = self.super_gru(h_super, context)
            h_super = nn.leaky_relu(h_super)
        return h_super

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        num_graphs = int(mx.max(batch_indices).item()) + 1
        h = nn.leaky_relu(self.node_proj(node_features))

        def step_fn(h_in: mx.array) -> mx.array:
            return self.attfp_step(
                edge_index, h_in, edge_features, training=training
            )

        graph_embeddings_list = []
        for _ in range(self.ode_steps):
            h = _ode_step_euler(h, self.ode_dt, step_fn)
            h = nn.leaky_relu(h)
            g_t = self._graph_embedding_from_atoms(
                h, batch_indices, num_graphs, training
            )
            graph_embeddings_list.append(g_t)

        G_stack = mx.stack(graph_embeddings_list, axis=1)
        scores = self.layer_attention(G_stack).squeeze(-1)
        attn_weights = mx.softmax(scores, axis=-1)
        mol_feature = mx.sum(attn_weights[:, :, None] * G_stack, axis=1)

        # AttentiveFP-style mol-level refinement (for fair comparison)
        if self.mol_attention_steps > 0 and self.mol_align is not None:
            for _ in range(self.mol_attention_steps):
                mol_expand = mol_feature[batch_indices]
                align_inp = mx.concatenate([mol_expand, h], axis=-1)
                align_score = nn.leaky_relu(self.mol_align(align_inp)).reshape(-1)
                max_per_graph = scatter(
                    align_score, batch_indices, out_size=num_graphs, aggr="max"
                )
                attention_weight = mx.exp(
                    align_score - max_per_graph[batch_indices]
                )
                norm = scatter(
                    attention_weight, batch_indices, out_size=num_graphs, aggr="add"
                )
                attention_weight = attention_weight / (norm[batch_indices] + 1e-8)
                atom_transform = self.mol_attend(
                    self.dropout(h) if training else h
                )
                mol_context = scatter(
                    attention_weight[:, None] * atom_transform,
                    batch_indices,
                    out_size=num_graphs,
                    aggr="add",
                )
                mol_context = nn.elu(mol_context)
                mol_feature = self.mol_gru(mol_feature, mol_context)
                mol_feature = nn.leaky_relu(mol_feature)

        if (
            self.rdkit_dim > 0
            and graph_features is not None
            and self.rdkit_norm is not None
        ):
            graph_features_norm = self.rdkit_norm(graph_features)
            r0 = mx.concatenate([mol_feature, graph_features_norm], axis=-1)
        else:
            r0 = mol_feature
        r0 = self.dropout(r0) if training else r0
        for i, layer in enumerate(self.mlp_layers):
            r0 = layer(r0)
            if i < len(self.mlp_layers) - 1:
                r0 = nn.leaky_relu(r0)
        return r0
