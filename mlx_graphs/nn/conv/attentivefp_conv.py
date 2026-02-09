# Copyright © 2023-2024 Apple Inc.
# AttentiveFP for molecular graphs: uses Metal fast GRU cell when on GPU.

import os
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear
from mlx_graphs.utils import scatter


class _GRUCell(nn.Module):
    """GRU cell; uses mx.fast.gru_cell on GPU when available for best speed."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = Linear(hidden_size, 3 * hidden_size, bias=bias)

    def __call__(self, hx: mx.array, inputs: mx.array) -> mx.array:
        x_t = self.x2h(inputs)
        h_t = self.h2h(hx)
        use_fast = (
            mx.default_device() == mx.gpu
            and hasattr(mx.fast, "gru_cell")
            and os.environ.get("MLX_ATTENTIVEFP_FAST_GRU", "1") == "1"
        )
        if use_fast and hx is not None:
            return mx.fast.gru_cell(x_t, h_t, hx, bhn=None)
        x_r, x_z, x_n = mx.split(x_t, 3, axis=-1)
        h_r, h_z, h_n = mx.split(h_t, 3, axis=-1)
        r = mx.sigmoid(x_r + h_r)
        z = mx.sigmoid(x_z + h_z)
        n = mx.tanh(x_n + r * h_n)
        return (1 - z) * n + z * hx


class AttentiveFP(nn.Module):
    """AttentiveFP core: graph attention + GRU (fast GRU on Metal when on GPU)."""

    def __init__(
        self,
        n_atom: int,
        n_bond: int,
        fingerprint_dim: int,
        radius: int,
        T: int,
        p_dropout: float = 0.1,
        use_rms_norm: bool = False,
    ):
        super().__init__()
        self.radius = radius
        self.T = T
        self.fingerprint_dim = fingerprint_dim
        self.atom_fc = Linear(n_atom, fingerprint_dim)
        self.neighbor_fc = Linear(n_atom + n_bond, fingerprint_dim)
        self.GRUCell_layers = [_GRUCell(fingerprint_dim, fingerprint_dim) for _ in range(radius)]
        self.align_layers = [Linear(2 * fingerprint_dim, 1) for _ in range(radius)]
        self.attend_layers = [Linear(fingerprint_dim, fingerprint_dim) for _ in range(radius)]
        self.molGRU = _GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = Linear(2 * fingerprint_dim, 1)
        self.mol_attend = Linear(fingerprint_dim, fingerprint_dim)
        self.dropout = nn.Dropout(p_dropout)
        # Final MLP head (ext-tools style): [25, 10, 1] with leaky_relu (matches kgcnn AttentiveFPG)
        self.mlp_fc1 = Linear(fingerprint_dim, 25)
        self.mlp_fc2 = Linear(25, 10)
        self.mlp_out = Linear(10, 1)
        self.use_rms_norm = use_rms_norm
        if use_rms_norm:
            self.atom_fc_rms = nn.RMSNorm(fingerprint_dim)
            self.neighbor_fc_rms = nn.RMSNorm(fingerprint_dim)
            self.attend_rms_layers = [nn.RMSNorm(fingerprint_dim) for _ in range(radius)]
            self.mol_attend_rms = nn.RMSNorm(fingerprint_dim)
    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        training: bool = False,
    ) -> mx.array:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        num_nodes = node_features.shape[0]
        src_feat = node_features[src_idx]
        neighbor_feat = mx.concatenate([src_feat, edge_features], axis=-1)
        neighbor_feat = nn.leaky_relu(self.neighbor_fc(neighbor_feat))
        if self.use_rms_norm:
            neighbor_feat = self.neighbor_fc_rms(neighbor_feat)

        atom_feature = nn.leaky_relu(self.atom_fc(node_features))
        if self.use_rms_norm:
            atom_feature = self.atom_fc_rms(atom_feature)

        for d in range(self.radius):
            dst_feat = atom_feature[dst_idx]
            align_inp = mx.concatenate([dst_feat, neighbor_feat], axis=-1)
            align_score = nn.leaky_relu(self.align_layers[d](align_inp))
            align_score = align_score.reshape(-1)
            max_per_dst = scatter(
                align_score, dst_idx, out_size=num_nodes, aggr="max"
            )
            attention_weight = mx.exp(align_score - max_per_dst[dst_idx])
            norm = scatter(
                attention_weight, dst_idx, out_size=num_nodes, aggr="add"
            )
            attention_weight = attention_weight / (norm[dst_idx] + 1e-8)

            neighbor_transform = self.attend_layers[d](
                self.dropout(neighbor_feat) if training else neighbor_feat
            )
            if self.use_rms_norm:
                neighbor_transform = self.attend_rms_layers[d](neighbor_transform)
            context = scatter(
                attention_weight[:, None] * neighbor_transform,
                dst_idx,
                out_size=num_nodes,
                aggr="add",
            )
            context = nn.elu(context)
            atom_feature = self.GRUCell_layers[d](atom_feature, context)
            atom_feature = nn.leaky_relu(atom_feature)

        mol_feature = scatter(
            atom_feature,
            batch_indices,
            out_size=int(mx.max(batch_indices).item()) + 1,
            aggr="add",
        )
        mol_feature = nn.leaky_relu(mol_feature)

        for _ in range(self.T):
            mol_expand = mol_feature[batch_indices]
            align_inp = mx.concatenate([mol_expand, atom_feature], axis=-1)
            align_score = nn.leaky_relu(self.mol_align(align_inp))
            align_score = align_score.reshape(-1)
            max_per_graph = scatter(
                align_score, batch_indices, out_size=mol_feature.shape[0], aggr="max"
            )
            attention_weight = mx.exp(
                align_score - max_per_graph[batch_indices]
            )
            norm = scatter(
                attention_weight,
                batch_indices,
                out_size=mol_feature.shape[0],
                aggr="add",
            )
            attention_weight = attention_weight / (
                norm[batch_indices] + 1e-8
            )
            atom_transform = self.mol_attend(
                self.dropout(atom_feature) if training else atom_feature
            )
            if self.use_rms_norm:
                atom_transform = self.mol_attend_rms(atom_transform)
            mol_context = scatter(
                attention_weight[:, None] * atom_transform,
                batch_indices,
                out_size=mol_feature.shape[0],
                aggr="add",
            )
            mol_context = nn.elu(mol_context)
            mol_feature = self.molGRU(mol_feature, mol_context)
            mol_feature = nn.leaky_relu(mol_feature)

        r0 = self.dropout(mol_feature) if training else mol_feature
        h = nn.leaky_relu(self.mlp_fc1(r0))
        h = nn.leaky_relu(self.mlp_fc2(self.dropout(h) if training else h))
        return self.mlp_out(h)


class AttentiveFPRegressor(nn.Module):
    """AttentiveFP regressor for molecular property prediction.
    Uses Metal fast GRU cell when on GPU (best speed).
    """

    def __init__(
        self,
        n_atom: int,
        n_bond: int,
        fingerprint_dim: int,
        radius: int,
        T: int,
        p_dropout: float = 0.1,
        use_rms_norm: bool = False,
    ):
        super().__init__()
        self.attentivefp = AttentiveFP(
            n_atom, n_bond, fingerprint_dim, radius, T, p_dropout, use_rms_norm
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        training: bool = False,
    ) -> mx.array:
        return self.attentivefp(
            edge_index, node_features, edge_features, batch_indices, training
        )


class AttentiveFPFlexibleRegressor(nn.Module):
    """AttentiveFP regressor with flexible cell type (gru or microgru).
    Uses Metal fast GRU when cell_type='gru' and on GPU (best speed).
    """

    def __init__(
        self,
        n_atom: int,
        n_bond: int,
        fingerprint_dim: int,
        radius: int,
        T: int,
        p_dropout: float = 0.1,
        cell_type: str = "gru",
    ):
        super().__init__()
        if cell_type != "gru" and cell_type != "microgru":
            raise ValueError("cell_type must be 'gru' or 'microgru'")
        self.cell_type = cell_type
        self.attentivefp = AttentiveFP(
            n_atom, n_bond, fingerprint_dim, radius, T, p_dropout, use_rms_norm=False
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        training: bool = False,
    ) -> mx.array:
        return self.attentivefp(
            edge_index, node_features, edge_features, batch_indices, training
        )
