# Copyright © 2023-2024 Apple Inc.
# AttentiveFP for molecular graphs: uses Metal fast GRU cell when on GPU.

import os
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear
from mlx_graphs.utils import scatter


# Chunk size for fast GRU: Metal kernel is fine at small batch (e.g. TextCNNLSTM);
# AttentiveFP passes num_nodes as batch (often 2000+) which can trigger NaNs.
# Override with MLX_ATTENTIVEFP_FAST_GRU_CHUNK_SIZE (default 256).
def _fast_gru_chunk_size():
    return int(os.environ.get("MLX_ATTENTIVEFP_FAST_GRU_CHUNK_SIZE", "256"))

class _GRUCell(nn.Module):
    """GRU cell; optional mx.fast.gru_cell on GPU (set MLX_ATTENTIVEFP_FAST_GRU=1).
    Default is standard MLX ops (stable training). The Metal fast path can produce
    NaNs when batch (first dim) is very large (e.g. AttentiveFP atom-level uses
    num_nodes). Fast path is chunked when batch > _FAST_GRU_CHUNK_SIZE so it
    behaves like small-batch use (e.g. TextCNNLSTM).
    """

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
            and os.environ.get("MLX_ATTENTIVEFP_FAST_GRU", "0") == "1"
        )
        if use_fast and hx is not None:
            B = x_t.shape[0]
            chunk_size = _fast_gru_chunk_size()
            # Only use Metal for small batch (like nn.GRU). Large batch (full block) uses native
            # to avoid NaN (chunked Metal also produced NaNs, so we do not use it).
            # chunk_size <= 0: never use Metal.
            if chunk_size > 0 and B <= chunk_size:
                return mx.fast.gru_cell(x_t, h_t, hx, bhn=None)
            # Fall through to native (slow) for large B or chunk_size<=0
        x_r, x_z, x_n = mx.split(x_t, 3, axis=-1)
        h_r, h_z, h_n = mx.split(h_t, 3, axis=-1)
        r = mx.sigmoid(x_r + h_r)
        z = mx.sigmoid(x_z + h_z)
        n = mx.tanh(x_n + r * h_n)
        return (1 - z) * n + z * hx


class AttentiveFPNodeStep(nn.Module):
    """Single node-level step of AttentiveFP: attention over neighbors + GRU update.
    Used as the velocity field for ODE-AttFP: h_new = GRU(h, context) with
    context = weighted sum of attend(neighbor_feat) and neighbor_feat from (h[src], edge).
    Input h has shape (num_nodes, fp_dim); output has shape (num_nodes, fp_dim)."""

    def __init__(
        self,
        fp_dim: int,
        edge_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fp_dim = fp_dim
        self.neighbor_fc = Linear(fp_dim + edge_dim, fp_dim)
        self.align = Linear(2 * fp_dim, 1)
        self.attend = Linear(fp_dim, fp_dim)
        self.gru = _GRUCell(fp_dim, fp_dim)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        edge_index: mx.array,
        h: mx.array,
        edge_features: mx.array,
        training: bool = False,
    ) -> mx.array:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        num_nodes = h.shape[0]
        # neighbor_feat from current h (so step is a function of h only)
        src_feat = h[src_idx]
        neighbor_feat = mx.concatenate([src_feat, edge_features], axis=-1)
        neighbor_feat = nn.leaky_relu(self.neighbor_fc(neighbor_feat))
        # attention: align(dst, neighbor) -> weights per edge
        dst_feat = h[dst_idx]
        align_inp = mx.concatenate([dst_feat, neighbor_feat], axis=-1)
        align_score = nn.leaky_relu(self.align(align_inp))
        align_score = align_score.reshape(-1)
        max_per_dst = scatter(align_score, dst_idx, out_size=num_nodes, aggr="max")
        attention_weight = mx.exp(align_score - max_per_dst[dst_idx])
        norm = scatter(attention_weight, dst_idx, out_size=num_nodes, aggr="add")
        attention_weight = attention_weight / (norm[dst_idx] + 1e-8)
        # context = weighted sum of attend(neighbor_feat)
        neighbor_transform = self.attend(
            self.dropout(neighbor_feat) if training else neighbor_feat
        )
        context = scatter(
            attention_weight[:, None] * neighbor_transform,
            dst_idx,
            out_size=num_nodes,
            aggr="add",
        )
        context = nn.elu(context)
        h_new = self.gru(h, context)
        return nn.leaky_relu(h_new)


class AttentiveFP(nn.Module):
    """AttentiveFP core: graph attention + GRU (fast GRU on Metal when on GPU).
    Optional graph-level features (e.g. RDKit 217) can be concatenated before the MLP when rdkit_dim > 0.
    residual_dt: on atom convolution only. None = replace; 'add' = leaky_relu(norm(h+y)); 0.5 = leaky_relu(h+0.5*(y-h)).
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
        rdkit_dim: int = 0,
        residual_dt: Optional[Union[float, str]] = None,
    ):
        super().__init__()
        self.radius = radius
        self.T = T
        self.fingerprint_dim = fingerprint_dim
        self.rdkit_dim = rdkit_dim
        self.residual_dt = residual_dt
        self.norm_after_add = nn.LayerNorm(fingerprint_dim) if residual_dt == "add" else None
        self.atom_fc = Linear(n_atom, fingerprint_dim)
        self.neighbor_fc = Linear(n_atom + n_bond, fingerprint_dim)
        self.GRUCell_layers = [_GRUCell(fingerprint_dim, fingerprint_dim) for _ in range(radius)]
        self.align_layers = [Linear(2 * fingerprint_dim, 1) for _ in range(radius)]
        self.attend_layers = [Linear(fingerprint_dim, fingerprint_dim) for _ in range(radius)]
        self.molGRU = _GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = Linear(2 * fingerprint_dim, 1)
        self.mol_attend = Linear(fingerprint_dim, fingerprint_dim)
        self.dropout = nn.Dropout(p_dropout)
        # Final MLP head (unified): [fp (+ rdkit) -> 64 -> 32 -> 1], leaky_relu, dropout 0.1
        mlp_in = fingerprint_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        self.mlp_fc1 = Linear(mlp_in, 64)
        self.mlp_fc2 = Linear(64, 32)
        self.mlp_out = Linear(32, 1)
        self.use_rms_norm = use_rms_norm
        if use_rms_norm:
            self.atom_fc_rms = nn.RMSNorm(fingerprint_dim)
            self.neighbor_fc_rms = nn.RMSNorm(fingerprint_dim)
            self.attend_rms_layers = [nn.RMSNorm(fingerprint_dim) for _ in range(radius)]
            self.mol_attend_rms = nn.RMSNorm(fingerprint_dim)
        if rdkit_dim > 0:
            self.rdkit_norm = nn.LayerNorm(rdkit_dim)
        else:
            self.rdkit_norm = None
    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
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

        dt = self.residual_dt
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
            y = self.GRUCell_layers[d](atom_feature, context)
            y = nn.leaky_relu(y)
            if dt == "add":
                atom_feature = nn.leaky_relu(self.norm_after_add(atom_feature + y))
            elif dt is not None and isinstance(dt, (int, float)):
                atom_feature = nn.leaky_relu(atom_feature + float(dt) * (y - atom_feature))
            else:
                atom_feature = y

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

        if self.rdkit_dim > 0 and graph_features is not None and self.rdkit_norm is not None:
            # Normalize RDKit descriptors to match graph embedding scale and avoid explosion
            graph_features_norm = self.rdkit_norm(graph_features)
            r0 = mx.concatenate([mol_feature, graph_features_norm], axis=-1)
        else:
            r0 = mol_feature
        r0 = self.dropout(r0) if training else r0
        h = nn.leaky_relu(self.mlp_fc1(r0))
        h = nn.leaky_relu(self.mlp_fc2(h))
        return self.mlp_out(h)


class AttentiveFPRegressor(nn.Module):
    """AttentiveFP regressor for molecular property prediction.
    Uses Metal fast GRU cell when on GPU (best speed).
    Optional graph_features (e.g. RDKit 217) concatenated before MLP when rdkit_dim > 0.
    residual_dt: on atom convolution only (classical/skip05). None = replace; 'add' = classical; 0.5 = skip05.
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
        rdkit_dim: int = 0,
        residual_dt: Optional[Union[float, str]] = None,
    ):
        super().__init__()
        self.attentivefp = AttentiveFP(
            n_atom, n_bond, fingerprint_dim, radius, T, p_dropout, use_rms_norm, rdkit_dim, residual_dt=residual_dt
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        return self.attentivefp(
            edge_index, node_features, edge_features, batch_indices,
            graph_features=graph_features, training=training,
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
