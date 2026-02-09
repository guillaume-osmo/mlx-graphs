# Step 2 (MLX): RNN-only commit — correct commands

You are on `rnn-speedup-only` (from `origin/main`). The RNN files **don't exist** on this branch yet — they exist only on `backup-all-my-work-feb2`. So you must **restore them from backup** first, then add and commit.

**Paths in the mlx repo:** use `mlx/` (not `mlx/mlx/`) — e.g. `mlx/backend/metal/fast_gru_cell.cpp`.

**Exclude FlashAttention:** do not restore `flash_attention.*`, `paged_attn/`, or primitives/export changes that are only for FlashAttention.

---

## 2a. Restore RNN-only files from backup

Run from repo root: `/Users/guillaume-osmo/Github/mlx`

```bash
cd /Users/guillaume-osmo/Github/mlx
git checkout backup-all-my-work-feb2 -- \
  python/mlx/nn/layers/RECURRENT_VERSIONS.md \
  python/mlx/nn/layers/recurrent.py \
  python/mlx/nn/layers/__init__.py \
  mlx/backend/metal/fast_gru_cell.cpp \
  mlx/backend/metal/fast_lstm_cell.cpp \
  mlx/backend/metal/kernels/fast_gru_cell.h \
  mlx/backend/metal/kernels/fast_gru_cell.metal \
  mlx/backend/metal/kernels/fast_lstm_cell.h \
  mlx/backend/metal/kernels/fast_lstm_cell.metal
```

Do **not** run:
- `git checkout backup-all-my-work-feb2 -- mlx/backend/metal/flash_attention.cpp` (or any `flash_attention*`, `paged_attn`).

---

## 2b. Restore “mixed” files (fast + Python bindings)

These files may contain **both** RNN (gru_cell, lstm_cell) and FlashAttention. To keep the commit **RNN-only**, you have two options.

### Option A — RNN-only (recommended)

Restore the four files from backup, then remove FlashAttention-related bits by hand (or with a small patch) so only RNN remains. Then add only these edited files.

### Option B — Restore as-is (commit will include FlashAttention in these files)

If you are okay with the commit containing both RNN and FlashAttention in `fast.cpp` / `fast.h` / `fast_primitives.h` / `python/src/fast.cpp`:

```bash
git checkout backup-all-my-work-feb2 -- \
  mlx/fast.cpp \
  mlx/fast.h \
  mlx/fast_primitives.h \
  python/src/fast.cpp
```

---

## 2c. CMakeLists — add only RNN kernels (no FlashAttention)

Do **not** restore the full `CMakeLists.txt` from backup (that would pull in FlashAttention build).

**1) Metal kernels CMakeLists**

```bash
git checkout backup-all-my-work-feb2 -- mlx/backend/metal/kernels/CMakeLists.txt
```

Then open `mlx/backend/metal/kernels/CMakeLists.txt` and **delete** any lines that mention `flash_attention` or `paged_attn`. Keep only the two lines for:

- `fast_gru_cell`
- `fast_lstm_cell`

**2) Metal CMakeLists**

```bash
git checkout backup-all-my-work-feb2 -- mlx/backend/metal/CMakeLists.txt
```

Then open `mlx/backend/metal/CMakeLists.txt` and **delete** any lines that add `flash_attention.cpp` (or similar). Keep only the lines that add:

- `fast_gru_cell.cpp`
- `fast_lstm_cell.cpp`

---

## 2d. Primitives and export (only if needed for RNN)

- **Do not** restore `mlx/backend/cuda/primitives.cpp`, `mlx/backend/no_gpu/primitives.cpp`, or `mlx/export.cpp` unless they contain **only** RNN (FastGruCell / FastLSTMCell).  
- If on your backup they only add RNN and no FlashAttention, you can:

  ```bash
  git checkout backup-all-my-work-feb2 -- \
    mlx/backend/cuda/primitives.cpp \
    mlx/backend/no_gpu/primitives.cpp \
    mlx/export.cpp
  ```

  If they also register FlashAttention, leave them as on `origin/main` or edit after checkout to remove FlashAttention.

---

## 2e. Add only RNN-related files and commit

```bash
git add \
  python/mlx/nn/layers/recurrent.py \
  python/mlx/nn/layers/RECURRENT_VERSIONS.md \
  python/mlx/nn/layers/__init__.py \
  mlx/fast.cpp \
  mlx/fast.h \
  mlx/fast_primitives.h \
  mlx/backend/metal/fast_gru_cell.cpp \
  mlx/backend/metal/fast_lstm_cell.cpp \
  mlx/backend/metal/kernels/fast_gru_cell.h \
  mlx/backend/metal/kernels/fast_gru_cell.metal \
  mlx/backend/metal/kernels/fast_lstm_cell.h \
  mlx/backend/metal/kernels/fast_lstm_cell.metal \
  mlx/backend/metal/kernels/CMakeLists.txt \
  mlx/backend/metal/CMakeLists.txt \
  python/src/fast.cpp
# Only if you restored them and they are RNN-only:
# git add mlx/backend/cuda/primitives.cpp mlx/backend/no_gpu/primitives.cpp mlx/export.cpp
```

Check:

```bash
git status
git diff --cached --name-only
```

Commit:

```bash
git commit -m "RNN speedup: Metal fast GRU/LSTM cells, MLX_RNN_IMPL, fixes for loaded models"
```

---

## Summary

| What | Action |
|------|--------|
| RNN-only new files | `git checkout backup-all-my-work-feb2 --` the list in 2a (correct paths: `mlx/...`, not `mlx/mlx/...`). |
| RECURRENT_VERSIONS.md | Comes from backup (it doesn’t exist on `origin/main`). |
| FlashAttention | Do not checkout `flash_attention.*`, `paged_attn/`. In CMakeLists, keep only fast_gru_cell / fast_lstm_cell. |
| fast.cpp / fast.h / fast_primitives.h / python/src/fast.cpp | Either restore from backup and then remove FlashAttention (Option A), or restore as-is (Option B). |
