# Copyright © 2023-2024 Apple Inc.
# Mol-level GRU/attention readout matching the authors' AttentiveFP mol (virtual node) steps.
# Reference: https://github.com/OpenDrugAI/AttentiveFP code/AttentiveFP/AttentiveLayers.py
# (initial sum pool, then T steps: concat(mol, atom) -> align score -> softmax per graph ->
# weighted sum of mol_attend(atoms) -> ELU -> GRU(mol, context) -> LeakyReLU).

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear
from mlx_graphs.utils import scatter


def _get_gru_cell():
    from mlx_graphs.nn.conv.attentivefp_conv import _GRUCell
    return _GRUCell


class MolAttentionReadout(nn.Module):
    """Mol-level readout (AttentiveFP-style): initial sum pool,
    then T steps of (align mol↔atoms, softmax per graph, weighted context, GRU, LeakyReLU).
    Input: node_features (num_nodes, hidden_dim), batch_indices. Output: (num_graphs, hidden_dim)."""

    def __init__(
        self,
        hidden_dim: int,
        num_steps: int = 2,
        dropout: float = 0.1,
        integrator: str = "euler",
    ):
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if integrator not in ("euler", "heun"):
            raise ValueError("integrator must be one of ('euler', 'heun')")
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.integrator = integrator
        self.mol_align = Linear(2 * hidden_dim, 1)
        self.mol_attend = Linear(hidden_dim, hidden_dim)
        _GRUCell = _get_gru_cell()
        self.mol_gru = _GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _mol_context(
        self,
        mol_feature: mx.array,
        node_features: mx.array,
        batch_indices: mx.array,
        num_graphs: int,
        training: bool,
    ) -> mx.array:
        mol_expand = mol_feature[batch_indices]
        align_inp = mx.concatenate([mol_expand, node_features], axis=-1)
        align_score = nn.leaky_relu(self.mol_align(align_inp)).reshape(-1)
        max_per_graph = scatter(
            align_score, batch_indices, out_size=num_graphs, aggr="max"
        )
        attention_weight = mx.exp(align_score - max_per_graph[batch_indices])
        norm = scatter(
            attention_weight,
            batch_indices,
            out_size=num_graphs,
            aggr="add",
        )
        attention_weight = attention_weight / (norm[batch_indices] + 1e-8)
        atom_transform = self.mol_attend(
            self.dropout(node_features) if training else node_features
        )
        mol_context = scatter(
            attention_weight[:, None] * atom_transform,
            batch_indices,
            out_size=num_graphs,
            aggr="add",
        )
        return nn.elu(mol_context)

    def _mol_update(self, mol_feature: mx.array, mol_context: mx.array) -> mx.array:
        return nn.leaky_relu(self.mol_gru(mol_feature, mol_context))

    def __call__(
        self,
        node_features: mx.array,
        batch_indices: mx.array,
        training: bool = False,
    ) -> mx.array:
        num_graphs = int(mx.max(batch_indices).item()) + 1
        mol_feature = scatter(
            node_features,
            batch_indices,
            out_size=num_graphs,
            aggr="add",
        )
        mol_feature = nn.leaky_relu(mol_feature)
        for _ in range(self.num_steps):
            mol_context = self._mol_context(
                mol_feature, node_features, batch_indices, num_graphs, training
            )
            if self.integrator == "euler":
                mol_feature = self._mol_update(mol_feature, mol_context)
            else:
                # Heun predictor-corrector on GRU state transition:
                # F(h)=GRU+activation, delta(h)=F(h)-h, h_next=h+0.5*(delta1+delta2).
                pred = self._mol_update(mol_feature, mol_context)
                mol_context_2 = self._mol_context(
                    pred, node_features, batch_indices, num_graphs, training
                )
                corr = self._mol_update(pred, mol_context_2)
                mol_feature = 0.5 * (corr + mol_feature)
        return mol_feature
