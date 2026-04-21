"""
RODE-SmilesX / RODE-TextCNN + MolAttFPPooling + RODE-MLP vs baselines SmilesX-MLP, TextCNN-MLP.
No SMILES augmentation. Single-layer + ODE skip throughout.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .ode_layers import ODESkipLayer, ODE_MLP
from .mol_attfp_pooling import MolAttFPPooling


def _embed_and_lstm(
    embedding: mx.array,
    x: mx.array,
    lstm: nn.LSTM,
    lstm_bwd: nn.LSTM | None,
    lstm_units: int,
):
    """x: (batch, seq) int. Return (batch, seq, hidden_dim) or (batch, seq, 2*hidden_dim) if bidir."""
    flat = x.reshape(-1)
    emb = mx.take(embedding, flat, axis=0)
    emb = emb.reshape(x.shape + (embedding.shape[-1],))  # (batch, seq, emb_dim)
    batch = x.shape[0]
    h0 = mx.zeros((batch, lstm_units))
    c0 = mx.zeros((batch, lstm_units))
    out_fwd, _ = lstm(emb, h0, c0)  # (batch, seq, H)
    if lstm_bwd is not None:
        out_bwd, _ = lstm_bwd(emb[:, ::-1, :], h0, c0)
        out_bwd = out_bwd[:, ::-1, :]
        out = mx.concatenate([out_fwd, out_bwd], axis=-1)
    else:
        out = out_fwd
    return out


class RODE_SmilesX_MLP(nn.Module):
    """(RODE-SmilesX) -> [MolAttFPPooling or mean pool] -> (RODE-MLP). Regression."""

    def __init__(
        self,
        vocab_size: int,
        maxlen: int,
        embedding_dim: int = 64,
        lstm_units: int = 64,
        n_ode_steps: int = 4,
        ode_dt: float = 0.25,
        pool_steps: int = 2,
        dropout: float = 0.2,
        use_mol_attfp_pooling: bool = True,
    ):
        super().__init__()
        self.maxlen = maxlen
        self.lstm_units = lstm_units
        self.use_mol_attfp_pooling = use_mol_attfp_pooling
        scale = 1.0 / (embedding_dim ** 0.5)
        self.embedding = mx.random.uniform(low=-scale, high=scale, shape=(vocab_size, embedding_dim))
        self.lstm_fwd = nn.LSTM(embedding_dim, lstm_units, bias=True)
        self.lstm_bwd = nn.LSTM(embedding_dim, lstm_units, bias=True)
        hidden_dim = 2 * lstm_units
        self.rode_seq = ODESkipLayer(hidden_dim, n_steps=n_ode_steps, dt=ode_dt)
        self.pool = MolAttFPPooling(hidden_dim, num_steps=pool_steps, dropout=dropout) if use_mol_attfp_pooling else None
        self.head = ODE_MLP(
            hidden_dim,
            128,
            output_dim=1,
            n_steps=3,
            dt=ode_dt,
            dropout=dropout,
            hidden_dims=(128, 64, 32),
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def __call__(self, x: mx.array, training: bool = True) -> mx.array:
        seq_enc = _embed_and_lstm(
            self.embedding, x, self.lstm_fwd, self.lstm_bwd, self.lstm_units
        )
        batch, seq_len, H = seq_enc.shape
        h = mx.reshape(seq_enc, (batch * seq_len, H))
        h = self.rode_seq(h)
        h = mx.reshape(h, (batch, seq_len, H))
        if self.dropout is not None and training:
            h = self.dropout(h)
        mol = self.pool(h, training=training) if self.pool is not None else mx.mean(h, axis=1)
        return self.head(mol, training=training)


class SmilesX_MLP(nn.Module):
    """SmilesX (BiLSTM) -> mean pool -> MLP. Baseline, no ODE."""

    def __init__(
        self,
        vocab_size: int,
        maxlen: int,
        embedding_dim: int = 64,
        lstm_units: int = 64,
        hidden_mlp: int = 128,
        dropout: float = 0.2,
        hash_mode: str = "linear",
        pool_mode: str = "mean",
    ):
        super().__init__()
        self.maxlen = maxlen
        self.hash_mode = hash_mode
        self.pool_mode = pool_mode
        if self.hash_mode not in ("linear", "glu"):
            raise ValueError(f"Unsupported hash_mode={self.hash_mode}")
        if self.pool_mode not in ("mean", "attn"):
            raise ValueError(f"Unsupported pool_mode={self.pool_mode}")
        scale = 1.0 / (embedding_dim ** 0.5)
        self.embedding = mx.random.uniform(low=-scale, high=scale, shape=(vocab_size, embedding_dim))
        self.lstm_fwd = nn.LSTM(embedding_dim, lstm_units, bias=True)
        self.lstm_bwd = nn.LSTM(embedding_dim, lstm_units, bias=True)
        self.lstm_units = lstm_units
        hidden_dim = 2 * lstm_units
        self.proj_v = nn.Linear(hidden_dim, hidden_dim)
        self.proj_g = nn.Linear(hidden_dim, hidden_dim) if self.hash_mode == "glu" else None
        self.attn = nn.Linear(hidden_dim, 1) if self.pool_mode == "attn" else None
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_mlp),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_mlp, 1),
        )

    def __call__(self, x: mx.array, training: bool = True) -> mx.array:
        seq_enc = _embed_and_lstm(
            self.embedding, x, self.lstm_fwd, self.lstm_bwd, self.lstm_units
        )
        seq_feat = self.proj_v(seq_enc)
        if self.proj_g is not None:
            seq_feat = seq_feat * nn.sigmoid(self.proj_g(seq_enc))
        if self.attn is not None:
            scores = self.attn(seq_feat)
            weights = mx.softmax(scores, axis=1)
            mol = mx.sum(weights * seq_feat, axis=1)
        else:
            mol = mx.mean(seq_feat, axis=1)
        return self.head(mol)


class RODE_TextCNN_MLP(nn.Module):
    """(RODE-TextCNN) -> [MolAttFPPooling or mean pool] -> (RODE-MLP). Regression."""

    def __init__(
        self,
        vocab_size: int,
        maxlen: int,
        embedding_dim: int = 64,
        conv_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        n_ode_steps: int = 4,
        ode_dt: float = 0.25,
        pool_steps: int = 2,
        dropout: float = 0.2,
        use_mol_attfp_pooling: bool = True,
        use_rode_cnn: bool = True,
    ):
        super().__init__()
        self.maxlen = maxlen
        self.use_mol_attfp_pooling = use_mol_attfp_pooling
        self.use_rode_cnn = use_rode_cnn
        scale = 1.0 / (embedding_dim ** 0.5)
        self.embedding = mx.random.uniform(low=-scale, high=scale, shape=(vocab_size, embedding_dim))
        self.convs = []
        self.paddings = []
        for i, k in enumerate(kernel_sizes):
            conv = nn.Conv1d(embedding_dim, conv_filters, kernel_size=k, padding=0)
            setattr(self, f"conv_{i}", conv)
            self.convs.append(conv)
            pl, pr = (k - 1) // 2, k - 1 - (k - 1) // 2
            self.paddings.append((pl, pr))
        hidden_dim = conv_filters * len(kernel_sizes)
        self.rode_seq = ODESkipLayer(hidden_dim, n_steps=n_ode_steps, dt=ode_dt) if use_rode_cnn else None
        self.pool = MolAttFPPooling(hidden_dim, num_steps=pool_steps, dropout=dropout) if use_mol_attfp_pooling else None
        self.head = ODE_MLP(
            hidden_dim,
            128,
            output_dim=1,
            n_steps=3,
            dt=ode_dt,
            dropout=dropout,
            hidden_dims=(128, 64, 32),
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def _conv_sequence(self, emb: mx.array) -> mx.array:
        conv_out = []
        for conv, (pl, pr) in zip(self.convs, self.paddings):
            if pl or pr:
                emb_pad = mx.pad(emb, [(0, 0), (pl, pr), (0, 0)])
            else:
                emb_pad = emb
            c = conv(emb_pad)
            c = nn.relu(c)
            conv_out.append(c)
        return mx.concatenate(conv_out, axis=-1)  # (batch, seq, conv_filters * n_kernels)

    def __call__(self, x: mx.array, training: bool = True) -> mx.array:
        flat = x.reshape(-1)
        emb = mx.take(self.embedding, flat, axis=0)
        emb = emb.reshape(x.shape + (self.embedding.shape[-1],))
        h = self._conv_sequence(emb)
        if self.rode_seq is not None:
            batch, seq_len, H = h.shape
            h = mx.reshape(h, (batch * seq_len, H))
            h = self.rode_seq(h)
            h = mx.reshape(h, (batch, seq_len, H))
        if self.dropout is not None and training:
            h = self.dropout(h)
        mol = self.pool(h, training=training) if self.pool is not None else mx.mean(h, axis=1)
        return self.head(mol, training=training)


class TextCNN_MLP(nn.Module):
    """TextCNN (Conv1d multi-kernel) -> mean pool -> MLP(128, 64, 32, 1). Baseline, no ODE."""

    def __init__(
        self,
        vocab_size: int,
        maxlen: int,
        embedding_dim: int = 64,
        conv_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        dropout: float = 0.2,
        hash_mode: str = "linear",
        pool_mode: str = "mean",
    ):
        super().__init__()
        self.maxlen = maxlen
        self.hash_mode = hash_mode
        self.pool_mode = pool_mode
        if self.hash_mode not in ("linear", "glu"):
            raise ValueError(f"Unsupported hash_mode={self.hash_mode}")
        if self.pool_mode not in ("mean", "attn"):
            raise ValueError(f"Unsupported pool_mode={self.pool_mode}")
        scale = 1.0 / (embedding_dim ** 0.5)
        self.embedding = mx.random.uniform(low=-scale, high=scale, shape=(vocab_size, embedding_dim))
        self.convs = []
        self.paddings = []
        for i, k in enumerate(kernel_sizes):
            conv = nn.Conv1d(embedding_dim, conv_filters, kernel_size=k, padding=0)
            setattr(self, f"conv_{i}", conv)
            self.convs.append(conv)
            pl, pr = (k - 1) // 2, k - 1 - (k - 1) // 2
            self.paddings.append((pl, pr))
        hidden_dim = conv_filters * len(kernel_sizes)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim)
        self.proj_g = nn.Linear(hidden_dim, hidden_dim) if self.hash_mode == "glu" else None
        self.attn = nn.Linear(hidden_dim, 1) if self.pool_mode == "attn" else None
        # MLP: 128 -> 64 -> 32 -> 1 (one dropout only, after first layer)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )

    def _conv_sequence(self, emb: mx.array) -> mx.array:
        conv_out = []
        for conv, (pl, pr) in zip(self.convs, self.paddings):
            emb_pad = mx.pad(emb, [(0, 0), (pl, pr), (0, 0)]) if (pl or pr) else emb
            c = nn.relu(conv(emb_pad))
            conv_out.append(c)
        return mx.concatenate(conv_out, axis=-1)

    def __call__(self, x: mx.array, training: bool = True) -> mx.array:
        flat = x.reshape(-1)
        emb = mx.take(self.embedding, flat, axis=0)
        emb = emb.reshape(x.shape + (self.embedding.shape[-1],))
        h = self._conv_sequence(emb)
        seq_feat = self.proj_v(h)
        if self.proj_g is not None:
            seq_feat = seq_feat * nn.sigmoid(self.proj_g(h))
        if self.attn is not None:
            scores = self.attn(seq_feat)
            weights = mx.softmax(scores, axis=1)
            mol = mx.sum(weights * seq_feat, axis=1)
        else:
            mol = mx.mean(seq_feat, axis=1)
        return self.head(mol)


def _get_pads(ksize: int) -> tuple[int, int]:
    return (int(np.ceil((ksize - 1) / 2)), (ksize - 1) // 2)


class CNF_MLP(nn.Module):
    """Convolutional Neural Fingerprint (CNF), paper flat variant (Kimber et al. 2018, arxiv.org/abs/1812.04439).
    No transformer: embed SMILES -> Layer 0 hash only (sum pool) + Layers 1..k Conv then sum pool -> concat -> MLP.
    Sum pooling as in paper ('sum the columns'); use with SMILES 10/10 augmentation for best results."""

    def __init__(
        self,
        vocab_size: int,
        maxlen: int,
        embedding_dim: int = 64,
        conv_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        dropout: float = 0.2,
        hash_mode: str = "linear",
        pool_mode: str = "sum",
    ):
        super().__init__()
        self.maxlen = maxlen
        self.hash_mode = hash_mode
        self.pool_mode = pool_mode
        if self.hash_mode not in ("linear", "glu"):
            raise ValueError(f"Unsupported hash_mode={self.hash_mode}")
        if self.pool_mode not in ("sum", "attn"):
            raise ValueError(f"Unsupported pool_mode={self.pool_mode}")
        scale = 1.0 / (embedding_dim ** 0.5)
        self.embedding = mx.random.uniform(low=-scale, high=scale, shape=(vocab_size, embedding_dim))
        self.convs = []
        self.paddings = []
        for i, k in enumerate(kernel_sizes):
            conv = nn.Conv1d(embedding_dim, conv_filters, kernel_size=k, padding=0)
            setattr(self, f"conv_{i}", conv)
            self.convs.append(conv)
            pl, pr = (k - 1) // 2, k - 1 - (k - 1) // 2
            self.paddings.append((pl, pr))
            setattr(self, f"proj_v_{i}", nn.Linear(conv_filters, conv_filters))
            if self.hash_mode == "glu":
                setattr(self, f"proj_g_{i}", nn.Linear(conv_filters, conv_filters))
            if self.pool_mode == "attn":
                setattr(self, f"attn_{i}", nn.Linear(conv_filters, 1))
        self.proj_hash_v = nn.Linear(embedding_dim, embedding_dim)
        self.proj_hash_g = nn.Linear(embedding_dim, embedding_dim) if self.hash_mode == "glu" else None
        self.attn_hash = nn.Linear(embedding_dim, 1) if self.pool_mode == "attn" else None
        # fingerprint dim: embed (hash-only branch) + conv_filters * len(kernel_sizes)
        hidden_dim = embedding_dim + conv_filters * len(kernel_sizes)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )

    def _conv_sequence(self, emb: mx.array) -> mx.array:
        conv_out = []
        for conv, (pl, pr) in zip(self.convs, self.paddings):
            emb_pad = mx.pad(emb, [(0, 0), (pl, pr), (0, 0)]) if (pl or pr) else emb
            c = nn.relu(conv(emb_pad))
            conv_out.append(c)
        return conv_out  # list of (batch, seq, conv_filters)

    def _project(self, x: mx.array, v_layer: nn.Linear, g_layer: nn.Linear | None) -> mx.array:
        v = v_layer(x)
        if g_layer is None:
            return v
        g = nn.sigmoid(g_layer(x))
        return v * g

    def _pool(self, x: mx.array, attn_layer: nn.Linear | None) -> mx.array:
        if attn_layer is None:
            return mx.sum(x, axis=1)
        scores = attn_layer(x)
        weights = mx.softmax(scores, axis=1)
        return mx.sum(weights * x, axis=1)

    def __call__(self, x: mx.array, training: bool = True) -> mx.array:
        flat = x.reshape(-1)
        emb = mx.take(self.embedding, flat, axis=0)
        emb = emb.reshape(x.shape + (self.embedding.shape[-1],))  # (batch, seq, emb_dim)
        hash_feat = self._project(emb, self.proj_hash_v, self.proj_hash_g)
        hash_pool = self._pool(hash_feat, self.attn_hash)
        conv_outs = self._conv_sequence(emb)
        conv_pools = []
        for i, c in enumerate(conv_outs):
            v_layer = getattr(self, f"proj_v_{i}")
            g_layer = getattr(self, f"proj_g_{i}") if self.hash_mode == "glu" else None
            attn_layer = getattr(self, f"attn_{i}") if self.pool_mode == "attn" else None
            feat = self._project(c, v_layer, g_layer)
            conv_pools.append(self._pool(feat, attn_layer))
        mol = mx.concatenate([hash_pool] + conv_pools, axis=-1)
        return self.head(mol)


class CNF_RODE_MLP(nn.Module):
    """CNF fingerprint (paper flat, no transformer) -> ODE-MLP head."""

    def __init__(
        self,
        vocab_size: int,
        maxlen: int,
        embedding_dim: int = 64,
        conv_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        ode_dt: float = 0.25,
        dropout: float = 0.2,
        hash_mode: str = "linear",
        pool_mode: str = "sum",
    ):
        super().__init__()
        self.maxlen = maxlen
        self.hash_mode = hash_mode
        self.pool_mode = pool_mode
        if self.hash_mode not in ("linear", "glu"):
            raise ValueError(f"Unsupported hash_mode={self.hash_mode}")
        if self.pool_mode not in ("sum", "attn"):
            raise ValueError(f"Unsupported pool_mode={self.pool_mode}")
        scale = 1.0 / (embedding_dim ** 0.5)
        self.embedding = mx.random.uniform(low=-scale, high=scale, shape=(vocab_size, embedding_dim))
        self.convs = []
        self.paddings = []
        for i, k in enumerate(kernel_sizes):
            conv = nn.Conv1d(embedding_dim, conv_filters, kernel_size=k, padding=0)
            setattr(self, f"conv_{i}", conv)
            self.convs.append(conv)
            pl, pr = (k - 1) // 2, k - 1 - (k - 1) // 2
            self.paddings.append((pl, pr))
            setattr(self, f"proj_v_{i}", nn.Linear(conv_filters, conv_filters))
            if self.hash_mode == "glu":
                setattr(self, f"proj_g_{i}", nn.Linear(conv_filters, conv_filters))
            if self.pool_mode == "attn":
                setattr(self, f"attn_{i}", nn.Linear(conv_filters, 1))
        self.proj_hash_v = nn.Linear(embedding_dim, embedding_dim)
        self.proj_hash_g = nn.Linear(embedding_dim, embedding_dim) if self.hash_mode == "glu" else None
        self.attn_hash = nn.Linear(embedding_dim, 1) if self.pool_mode == "attn" else None
        hidden_dim = embedding_dim + conv_filters * len(kernel_sizes)
        self.head = ODE_MLP(
            hidden_dim,
            128,
            output_dim=1,
            n_steps=3,
            dt=ode_dt,
            dropout=dropout,
            hidden_dims=(128, 64, 32),
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def _conv_sequence(self, emb: mx.array) -> mx.array:
        conv_out = []
        for conv, (pl, pr) in zip(self.convs, self.paddings):
            emb_pad = mx.pad(emb, [(0, 0), (pl, pr), (0, 0)]) if (pl or pr) else emb
            c = nn.relu(conv(emb_pad))
            conv_out.append(c)
        return conv_out

    def _project(self, x: mx.array, v_layer: nn.Linear, g_layer: nn.Linear | None) -> mx.array:
        v = v_layer(x)
        if g_layer is None:
            return v
        g = nn.sigmoid(g_layer(x))
        return v * g

    def _pool(self, x: mx.array, attn_layer: nn.Linear | None) -> mx.array:
        if attn_layer is None:
            return mx.sum(x, axis=1)
        scores = attn_layer(x)
        weights = mx.softmax(scores, axis=1)
        return mx.sum(weights * x, axis=1)

    def __call__(self, x: mx.array, training: bool = True) -> mx.array:
        flat = x.reshape(-1)
        emb = mx.take(self.embedding, flat, axis=0)
        emb = emb.reshape(x.shape + (self.embedding.shape[-1],))
        hash_feat = self._project(emb, self.proj_hash_v, self.proj_hash_g)
        hash_pool = self._pool(hash_feat, self.attn_hash)
        conv_outs = self._conv_sequence(emb)
        conv_pools = []
        for i, c in enumerate(conv_outs):
            v_layer = getattr(self, f"proj_v_{i}")
            g_layer = getattr(self, f"proj_g_{i}") if self.hash_mode == "glu" else None
            attn_layer = getattr(self, f"attn_{i}") if self.pool_mode == "attn" else None
            feat = self._project(c, v_layer, g_layer)
            conv_pools.append(self._pool(feat, attn_layer))
        mol = mx.concatenate([hash_pool] + conv_pools, axis=-1)
        if self.dropout is not None and training:
            mol = self.dropout(mol)
        return self.head(mol, training=training)


class CNF_MLP_NFP(nn.Module):
    """CNF logic as in cnf2/net.py (paper logic applied to transformer encoder). Not the paper's raw CNF.
    NFP + residual alpha + Batch_VM (max pool). Use CNF_MLP for the paper variant without transformer."""

    def __init__(
        self,
        vocab_size: int,
        maxlen: int,
        embedding_dim: int = 64,
        n_filters: int = 5,
        fp_dim: int = 64,
        mlp_dim: int = 128,
        kernel_sizes: tuple[int, ...] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.fp_dim = fp_dim
        if kernel_sizes is not None:
            self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = tuple(range(1, n_filters + 1))  # 1, 2, ..., n_filters
        n_filters = len(self.kernel_sizes)
        scale = 1.0 / (embedding_dim ** 0.5)
        self.embedding = mx.random.uniform(low=-scale, high=scale, shape=(vocab_size, embedding_dim))
        # Convs: in/out = embedding_dim (like net.py embdim)
        self.convs = []
        self.paddings = []
        for i, k in enumerate(self.kernel_sizes):
            conv = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=k, padding=0)
            setattr(self, f"conv_{i}", conv)
            self.convs.append(conv)
            pl, pr = _get_pads(k)
            self.paddings.append((pl, pr))
        # Alpha per layer (residual mix), init 0.5 (trainable via buffer; MLX optim will update if in parameters)
        self.alpha = mx.full((n_filters,), 0.5)
        # Batch_VM per branch: linear(embdim -> fp_dim) + activation + max pool (nn.Linear so trainable)
        for i in range(n_filters):
            setattr(self, f"H_{i}", nn.Linear(embedding_dim, fp_dim))
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else None
        outdim = n_filters * fp_dim
        self.head = nn.Sequential(
            nn.Linear(outdim, mlp_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_dim, 1),
        )

    def _batch_vm(self, x: mx.array, H: nn.Linear) -> mx.array:
        """x: (batch, seq, emb_dim). H: Linear(emb_dim, fp_dim). Out: (batch, fp_dim) via linear then max pool."""
        x = H(x)  # (batch, seq, fp_dim)
        x = nn.relu(x)
        return mx.max(x, axis=1)

    def __call__(self, x: mx.array, training: bool = True) -> mx.array:
        flat = x.reshape(-1)
        emb = mx.take(self.embedding, flat, axis=0)
        emb = emb.reshape(x.shape + (self.embedding.shape[-1],))  # (batch, seq, emb_dim)
        x = emb
        xcat = []
        x_cur = x
        for i in range(len(self.convs)):
            pl, pr = self.paddings[i]
            conv = getattr(self, f"conv_{i}")
            a = float(self.alpha[i])
            inp = a * x_cur + (1.0 - a) * x
            if pl or pr:
                inp = mx.pad(inp, [(0, 0), (pl, pr), (0, 0)])
            x_cur = nn.relu(conv(inp))
            H_layer = getattr(self, f"H_{i}")
            xcat.append(self._batch_vm(x_cur, H_layer))
        mol = mx.concatenate(xcat, axis=-1)
        if self.drop is not None and training:
            mol = self.drop(mol)
        return self.head(mol)
