# Commit only RNN speedup — safe steps (no code lost)

This guide lets you:
1. **Keep all your work** (nothing lost).
2. **Commit only RNN-related changes** in both repos.
3. **Know what to compare** with `ml-explore/mlx` origin.

---

## Step 0: Backup everything (do this first)

### MLX repo
```bash
cd /Users/guillaume-osmo/Github/mlx
git status
git branch backup-all-my-work    # create branch with current state
git checkout backup-all-my-work
git add -A
git commit -m "backup: all my local work before RNN-only cleanup"
git checkout main                  # back to main (or your dev branch)
```
Now your full state is saved on `backup-all-my-work`. You can always `git checkout backup-all-my-work` to get it back.

### mlx-graphs repo
```bash
cd /Users/guillaume-osmo/Github/mlx-graphs
git branch backup-all-my-work
git checkout backup-all-my-work
git add -A
git commit -m "backup: all my local work before RNN-only cleanup"
git checkout main
```

---

## Step 1: What is “RNN only” vs “not RNN”

### MLX repo — RNN speedup (core, commit these)

| Path (from mlx repo root) | Purpose |
|---------------------------|--------|
| `python/mlx/nn/layers/recurrent.py` | MLX_RNN_IMPL, fast path, eval/contiguous fixes |
| `python/mlx/nn/layers/RECURRENT_VERSIONS.md` | Doc legacy/fast |
| `python/mlx/nn/layers/__init__.py` | Export `get_rnn_implementation`, GRU, LSTM |
| `mlx/mlx/fast.cpp` | `gru_cell`, `lstm_cell` |
| `mlx/mlx/fast.h` | Declarations |
| `mlx/mlx/fast_primitives.h` | FastGruCell, FastLSTMCell |
| `mlx/mlx/backend/metal/fast_gru_cell.cpp` | Metal GRU kernel dispatch + always-copy fix |
| `mlx/mlx/backend/metal/fast_lstm_cell.cpp` | Metal LSTM kernel dispatch + output alloc fix |
| `mlx/mlx/backend/metal/kernels/fast_gru_cell.h` | GRU Metal kernel |
| `mlx/mlx/backend/metal/kernels/fast_gru_cell.metal` | GRU Metal source |
| `mlx/mlx/backend/metal/kernels/fast_lstm_cell.h` | LSTM Metal kernel |
| `mlx/mlx/backend/metal/kernels/fast_lstm_cell.metal` | LSTM Metal source |
| `mlx/mlx/backend/metal/kernels/CMakeLists.txt` | Build fast_gru_cell, fast_lstm_cell |
| `mlx/mlx/backend/metal/CMakeLists.txt` | Metal lib: fast_gru_cell.cpp, fast_lstm_cell.cpp |
| `python/src/fast.cpp` | Python bindings for `mx.fast.gru_cell`, `mx.fast.lstm_cell` |

### MLX repo — not RNN (do not add to RNN commit)

- Root-level scripts: `test_gru.py`, `benchmark_gru.py`, `compare_gru_implementations.py`, `debug_gru.py`, `mlx_bidirectional_gru.py`, etc. (your local tests/debug).
- Anything under `benchmarks/`, `examples/`, `tests/` that is not part of upstream and not RNN.
- Any other file you changed for non-RNN work.

### mlx-graphs repo — RNN-related (commit these)

| Path | Purpose |
|------|--------|
| `verify_fast_determinism.py` | FAST save == FAST run |
| `verify_fast_vs_legacy.py` | FAST vs LEGACY on loaded ref |
| `verify_recurrent_math.py` | Equations check |
| `compare_recurrent_outputs.py` | legacy vs fast I/O |
| `compare_recurrent_versions.py` | Speed legacy vs fast |
| `benchmark_attentivefp_and_gru.py` | Bench --layers, MLX_RNN_IMPL |
| `repro_fast_rnn_segfault.py` | Segfault repro |
| `GRU_LOADED_DATA_FIX.md` | Why GRU + loaded data fix |
| `SEGFAULT_FAST_RNN.md` | Segfault notes |
| `BENCHMARK_VS_COMPARE_SEGFAULT.md` | Optional |
| `FLOAT4_COMPATIBILITY_LOAD.md` | Optional |
| `docs/fast_gru_implementation.md` | Optional doc |

### mlx-graphs — not RNN (do not add to RNN commit)

- `benchmark_fast_gru.py` if it’s about a different GRU (e.g. custom conv), not `nn.GRU` speedup.
- `fix_gru.py`, `fix_gru_file.py`, `python/mlx/nn/gru.py` (custom GRU in mlx-graphs), unless you want them in this PR.
- Any other non-RNN changes.

---

## Step 2: Create RNN-only branch and commit (MLX)

```bash
cd /Users/guillaume-osmo/Github/mlx
git fetch origin
git checkout -b rnn-speedup-only origin/main
# If you develop on main and want to base on your main:
# git checkout -b rnn-speedup-only main
```

Then add **only** the RNN core files (adjust list if some paths don’t exist):

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
```

Check what will be committed (no non-RNN files):

```bash
git status
git diff --cached --name-only
```

Commit:

```bash
git commit -m "RNN speedup: Metal fast GRU/LSTM cells, MLX_RNN_IMPL, fixes for loaded models"
```

Your other changes stay in the working tree or on `backup-all-my-work`; they are not lost.

---

## Step 3: Create RNN-only branch and commit (mlx-graphs)

```bash
cd /Users/guillaume-osmo/Github/mlx-graphs
git fetch origin
git checkout -b rnn-speedup-only origin/main
# Or: git checkout -b rnn-speedup-only main
```

Add only RNN-related files:

```bash
git add \
  verify_fast_determinism.py \
  verify_fast_vs_legacy.py \
  verify_recurrent_math.py \
  compare_recurrent_outputs.py \
  compare_recurrent_versions.py \
  benchmark_attentivefp_and_gru.py \
  repro_fast_rnn_segfault.py \
  GRU_LOADED_DATA_FIX.md \
  SEGFAULT_FAST_RNN.md
# Optional:
# git add BENCHMARK_VS_COMPARE_SEGFAULT.md FLOAT4_COMPATIBILITY_LOAD.md docs/fast_gru_implementation.md
```

Check and commit:

```bash
git status
git diff --cached --name-only
git commit -m "RNN: verification and benchmark scripts for fast vs legacy GRU/LSTM"
```

---

## Step 4: Compare with origin MLX (what’s missing upstream)

```bash
cd /Users/guillaume-osmo/Github/mlx
git remote -v   # ensure 'origin' is ml-explore/mlx
git diff origin/main --name-only
# Or diff only RNN paths:
git diff origin/main -- python/mlx/nn/layers/recurrent.py mlx/mlx/backend/metal/fast_gru_cell.cpp ...
```

- If a file appears in `git diff origin/main` and is in the “RNN core” list above, that’s your RNN work that origin doesn’t have.
- If you see other changed files, they are non-RNN; don’t include them in the RNN-only commit.

---

## Step 5: Push and open PR (optional)

```bash
# MLX
cd /Users/guillaume-osmo/Github/mlx
git push -u origin rnn-speedup-only
# Then open PR: rnn-speedup-only → ml-explore/mlx main

# mlx-graphs
cd /Users/guillaume-osmo/Github/mlx-graphs
git push -u origin rnn-speedup-only
# Then open PR to mlx-graphs main
```

---

## If something goes wrong

- To restore your full state:  
  `git checkout backup-all-my-work`
- To see what’s only on backup:  
  `git log main..backup-all-my-work`
- Your uncommitted or other-branch work is not deleted by creating `rnn-speedup-only` or by the RNN-only commit; it remains in the repo or on `backup-all-my-work`.

---

## Quick checklist

- [ ] Backup branch created in both repos (`backup-all-my-work`).
- [ ] New branch `rnn-speedup-only` created from `origin/main` (or your `main`).
- [ ] Only RNN files added (lists above).
- [ ] `git status` and `git diff --cached --name-only` checked before commit.
- [ ] One commit per repo with a clear RNN-only message.
