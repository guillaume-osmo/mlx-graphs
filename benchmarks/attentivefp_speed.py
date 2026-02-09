# Copyright © 2023-2024 Apple Inc.
# Speed evaluation for AttentiveFP (AttentiveFPRegressor).
# Run from repo root: python benchmarks/attentivefp_speed.py [--device gpu|cpu] [--batch_size N]

import argparse
import time

import mlx.core as mx

from mlx_graphs.nn.conv import AttentiveFPRegressor


def make_batch(n_atom, n_bond, num_nodes, num_edges, batch_size, seed=0):
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


def main():
    p = argparse.ArgumentParser(description="AttentiveFP speed evaluation")
    p.add_argument(
        "--device",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Device to run on",
    )
    p.add_argument("--batch_size", type=int, default=32, help="Batch size (graphs)")
    p.add_argument("--num_nodes", type=int, default=512, help="Total nodes per batch")
    p.add_argument("--num_edges", type=int, default=2048, help="Total edges per batch")
    p.add_argument("--n_atom", type=int, default=32, help="Atom feature dim")
    p.add_argument("--n_bond", type=int, default=8, help="Bond feature dim")
    p.add_argument("--fp_dim", type=int, default=128, help="Fingerprint dim")
    p.add_argument("--radius", type=int, default=2, help="Radius")
    p.add_argument("--T", type=int, default=2, help="T (mol attention steps)")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    p.add_argument("--repeat", type=int, default=5, help="Timed runs (median taken)")
    args = p.parse_args()

    device = mx.gpu if args.device == "gpu" else mx.cpu
    mx.set_default_device(device)

    model = AttentiveFPRegressor(
        args.n_atom,
        args.n_bond,
        args.fp_dim,
        args.radius,
        args.T,
        p_dropout=0.1,
        use_rms_norm=True,
    )
    edge_index, node_features, edge_features, batch_indices = make_batch(
        args.n_atom,
        args.n_bond,
        args.num_nodes,
        args.num_edges,
        args.batch_size,
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

    for _ in range(args.warmup):
        forward()

    times_ms = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        forward()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]

    print("AttentiveFP speed evaluation")
    print(f"  device={args.device}  batch_size={args.batch_size}")
    print(f"  num_nodes={args.num_nodes}  num_edges={args.num_edges}")
    print(f"  n_atom={args.n_atom}  n_bond={args.n_bond}  fp_dim={args.fp_dim}")
    print(f"  radius={args.radius}  T={args.T}")
    print(f"  median forward time: {median_ms:.3f} ms")
    print(f"  throughput:         {args.batch_size / (median_ms / 1000):.1f} graphs/s")


if __name__ == "__main__":
    main()
