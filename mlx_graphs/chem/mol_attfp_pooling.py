"""
MolAttFP-style pooling for sequences: (batch, seq_len, hidden_dim) -> (batch, hidden_dim).
T steps: align mol with each position, softmax, weighted context, update mol (residual/GRU).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class MolAttFPPooling(nn.Module):
    """
    AttentiveFP-style readout for sequences (no graph structure).
    Input: (batch, seq_len, hidden_dim). Output: (batch, hidden_dim).
    Steps: mol = initial pool; for t in steps: align(mol, x) -> softmax -> weighted sum -> update mol.
    """

    def __init__(self, hidden_dim: int, num_steps: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.mol_align = nn.Linear(2 * hidden_dim, 1)
        self.mol_attend = nn.Linear(hidden_dim, hidden_dim)
        self.mol_update = nn.Linear(2 * hidden_dim, hidden_dim)  # residual: mol + f(mol, context)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        # x: (batch, seq_len, hidden_dim)
        batch, seq_len, H = x.shape
        mol = mx.mean(x, axis=1)  # (batch, H)
        mol = nn.leaky_relu(mol)
        for _ in range(self.num_steps):
            mol_expand = mx.expand_dims(mol, axis=1)  # (batch, 1, H)
            mol_expand = mx.broadcast_to(mol_expand, (batch, seq_len, H))
            align_inp = mx.concatenate([mol_expand, x], axis=-1)  # (batch, seq_len, 2*H)
            align_score = self.mol_align(align_inp)  # (batch, seq_len, 1)
            align_score = mx.squeeze(align_score, axis=-1)  # (batch, seq_len)
            attention_weight = mx.softmax(align_score, axis=-1)
            if training and self.dropout is not None:
                atom_transform = self.mol_attend(self.dropout(x))
            else:
                atom_transform = self.mol_attend(x)
            # weighted sum: (batch, seq_len, 1) * (batch, seq_len, H) -> sum over seq -> (batch, H)
            context = mx.sum(
                mx.expand_dims(attention_weight, axis=-1) * atom_transform,
                axis=1,
            )
            context = nn.elu(context)
            combined = mx.concatenate([mol, context], axis=-1)
            mol = mol + nn.leaky_relu(self.mol_update(combined))
        return mol
