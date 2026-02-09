# Copyright © 2023-2024 Apple Inc.
# Compare AttentiveFP speed: legacy GRU (Python) vs fast GRU (Metal).
# Run from repo root: python benchmarks/compare_attentivefp_legacy_fast.py [--device gpu]

import argparse
import os
import subprocess
import sys


def make_batch(n_atom, n_bond, num_nodes, num_edges, batch_size, seed=0):
    import mlx.core as mx

    mx.random.seed(seed)
    src = mx.random.randint(0, num_nodes, (num_edges,))
    dst = mx.random.randint(0, num_nodes, (num_edges,))
    edge_index = mx.stack([src, dst], axis=0)
    node_features = mx.random.normal((num_nodes, n_atom)).astype(mx.float32)
    edge_features = mx.random.normal((num_edges, n_bond)).astype(mx.float32)
    nodes_per_graph = max(1, (num_nodes + batch_size - 1) // batch_size)
    batch_indices = mx.minimum(
        mx.arange(num_nodes, dtype=mx.int32) // nodes_per_graph,
        batch_size - 1,
    )
    return edge_index, node_features, edge_features, batch_indices


def run_forward(device, fast_gru, warmup=10, repeat=5, **kwargs):
    import time

    import mlx.core as mx
    from mlx_graphs.nn.conv import AttentiveFPRegressor

    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)

    model = AttentiveFPRegressor(
        kwargs.get("n_atom", 32),
        kwargs.get("n_bond", 8),
        kwargs.get("fp_dim", 128),
        kwargs.get("radius", 2),
        kwargs.get("T", 2),
        p_dropout=0.1,
        use_rms_norm=True,
    )
    edge_index, node_features, edge_features, batch_indices = make_batch(
        kwargs.get("n_atom", 32),
        kwargs.get("n_bond", 8),
        kwargs.get("num_nodes", 512),
        kwargs.get("num_edges", 2048),
        kwargs.get("batch_size", 32),
    )

    def forward():
        out = model(
            edge_index,
            node_features,
            edge_features,
            batch_indices,
            training=False,
        )
        mx.eval(out)
        return out

    # Force the chosen GRU path by setting env before first call
    prev = os.environ.get("MLX_ATTENTIVEFP_FAST_GRU")
    try:
        os.environ["MLX_ATTENTIVEFP_FAST_GRU"] = "1" if fast_gru else "0"
        for _ in range(warmup):
            forward()
        times_ms = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            forward()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)
    finally:
        if prev is None:
            os.environ.pop("MLX_ATTENTIVEFP_FAST_GRU", None)
        else:
            os.environ["MLX_ATTENTIVEFP_FAST_GRU"] = prev

    times_ms.sort()
    return times_ms[len(times_ms) // 2]


def main():
    p = argparse.ArgumentParser(
        description="Compare AttentiveFP: legacy vs fast GRU speed"
    )
    p.add_argument(
        "--device",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Device (default: gpu)",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_nodes", type=int, default=512)
    p.add_argument("--num_edges", type=int, default=2048)
    p.add_argument("--n_atom", type=int, default=32)
    p.add_argument("--n_bond", type=int, default=8)
    p.add_argument("--fp_dim", type=int, default=128)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--T", type=int, default=2)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=5)
    args = p.parse_args()

    kwargs = {
        "batch_size": args.batch_size,
        "num_nodes": args.num_nodes,
        "num_edges": args.num_edges,
        "n_atom": args.n_atom,
        "n_bond": args.n_bond,
        "fp_dim": args.fp_dim,
        "radius": args.radius,
        "T": args.T,
        "warmup": args.warmup,
        "repeat": args.repeat,
    }

    print("AttentiveFP legacy vs fast GRU speed")
    print(f"  device={args.device}  batch_size={args.batch_size}")
    print(f"  num_nodes={args.num_nodes}  num_edges={args.num_edges}")
    print(f"  fp_dim={args.fp_dim}  radius={args.radius}  T={args.T}")
    print(f"  median of {args.repeat} runs (warmup={args.warmup})")
    print()

    legacy_ms = run_forward(args.device, fast_gru=False, **kwargs)
    fast_ms = run_forward(args.device, fast_gru=True, **kwargs)

    print("=" * 60)
    print("  Mode              Median (ms)")
    print("=" * 60)
    print(f"  legacy GRU (Python)   {legacy_ms:.3f}")
    print(f"  fast GRU (Metal)      {fast_ms:.3f}")
    print("=" * 60)
    if fast_ms > 0:
        speedup = legacy_ms / fast_ms
        print(f"  Speedup (fast vs legacy): {speedup:.2f}x")
    print()
    print("  Legacy: set MLX_ATTENTIVEFP_FAST_GRU=0 to force Python GRU.")
    print("  Fast:   default; uses mx.fast.gru_cell on GPU when available.")


if __name__ == "__main__":
    main()
