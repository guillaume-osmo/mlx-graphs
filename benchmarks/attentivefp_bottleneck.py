# Copyright © 2023-2024 Apple Inc.
# Full speed / bottleneck analysis for AttentiveFP: times each step to find limits.
# Run: python benchmarks/attentivefp_bottleneck.py [--device gpu] [--batch_size 32] [--repeat 5]

import argparse
import time

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv import AttentiveFPRegressor
from mlx_graphs.utils import scatter


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


def run_forward_with_timing(model, edge_index, node_features, edge_features, batch_indices):
    """Run one forward pass and return dict of step_name -> elapsed_ms (with mx.eval per step)."""
    m = model.attentivefp
    src_idx, dst_idx = edge_index[0], edge_index[1]
    num_nodes = node_features.shape[0]
    times = {}

    def eval_and_time(name, *arrays):
        t0 = time.perf_counter()
        mx.eval(*arrays)
        t1 = time.perf_counter()
        times[name] = (t1 - t0) * 1000

    # --- Initial projections ---
    src_feat = node_features[src_idx]
    neighbor_feat = mx.concatenate([src_feat, edge_features], axis=-1)
    neighbor_feat = nn.leaky_relu(m.neighbor_fc(neighbor_feat))
    if m.use_rms_norm:
        neighbor_feat = m.neighbor_fc_rms(neighbor_feat)
    eval_and_time("1_neighbor_fc", neighbor_feat)

    atom_feature = nn.leaky_relu(m.atom_fc(node_features))
    if m.use_rms_norm:
        atom_feature = m.atom_fc_rms(atom_feature)
    eval_and_time("2_atom_fc", atom_feature)

    # --- Radius steps ---
    for d in range(m.radius):
        dst_feat = atom_feature[dst_idx]
        align_inp = mx.concatenate([dst_feat, neighbor_feat], axis=-1)
        align_score = nn.leaky_relu(m.align_layers[d](align_inp))
        align_score = align_score.reshape(-1)
        max_per_dst = scatter(align_score, dst_idx, out_size=num_nodes, aggr="max")
        attention_weight = mx.exp(align_score - max_per_dst[dst_idx])
        norm = scatter(attention_weight, dst_idx, out_size=num_nodes, aggr="add")
        attention_weight = attention_weight / (norm[dst_idx] + 1e-8)
        eval_and_time(f"3_radius{d}_align_scatter", max_per_dst, norm, attention_weight)

        neighbor_transform = m.attend_layers[d](neighbor_feat)
        if m.use_rms_norm:
            neighbor_transform = m.attend_rms_layers[d](neighbor_transform)
        context = scatter(
            attention_weight[:, None] * neighbor_transform,
            dst_idx,
            out_size=num_nodes,
            aggr="add",
        )
        context = nn.elu(context)
        eval_and_time(f"4_radius{d}_attend_scatter", context)

        atom_feature = m.GRUCell_layers[d](atom_feature, context)
        atom_feature = nn.leaky_relu(atom_feature)
        eval_and_time(f"5_radius{d}_GRU", atom_feature)

    # --- Mol pool ---
    mol_feature = scatter(
        atom_feature,
        batch_indices,
        out_size=int(mx.max(batch_indices).item()) + 1,
        aggr="add",
    )
    mol_feature = nn.leaky_relu(mol_feature)
    eval_and_time("6_mol_pool", mol_feature)

    # --- T steps ---
    for t in range(m.T):
        mol_expand = mol_feature[batch_indices]
        align_inp = mx.concatenate([mol_expand, atom_feature], axis=-1)
        align_score = nn.leaky_relu(m.mol_align(align_inp))
        align_score = align_score.reshape(-1)
        max_per_graph = scatter(
            align_score, batch_indices, out_size=mol_feature.shape[0], aggr="max"
        )
        attention_weight = mx.exp(align_score - max_per_graph[batch_indices])
        norm = scatter(
            attention_weight, batch_indices, out_size=mol_feature.shape[0], aggr="add"
        )
        attention_weight = attention_weight / (norm[batch_indices] + 1e-8)
        eval_and_time(f"7_molT{t}_align_scatter", max_per_graph, norm, attention_weight)

        atom_transform = m.mol_attend(atom_feature)
        if m.use_rms_norm:
            atom_transform = m.mol_attend_rms(atom_transform)
        mol_context = scatter(
            attention_weight[:, None] * atom_transform,
            batch_indices,
            out_size=mol_feature.shape[0],
            aggr="add",
        )
        mol_context = nn.elu(mol_context)
        eval_and_time(f"8_molT{t}_attend_scatter", mol_context)

        mol_feature = m.molGRU(mol_feature, mol_context)
        mol_feature = nn.leaky_relu(mol_feature)
        eval_and_time(f"9_molT{t}_GRU", mol_feature)

    # --- MLP ---
    r0 = mol_feature
    r1 = nn.leaky_relu(m.linear1(r0))
    if m.use_rms_norm:
        r1 = m.linear1_rms(r1)
    r1 = r1 + r0
    eval_and_time("10_linear1", r1)

    r2 = nn.leaky_relu(m.linear2(r1))
    if m.use_rms_norm:
        r2 = m.linear2_rms(r2)
    r2 = r2 + r1
    out = m.output(r2)
    eval_and_time("11_linear2_output", out)

    return times


def main():
    p = argparse.ArgumentParser(description="AttentiveFP bottleneck / speed analysis")
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_nodes", type=int, default=512)
    p.add_argument("--num_edges", type=int, default=2048)
    p.add_argument("--n_atom", type=int, default=32)
    p.add_argument("--n_bond", type=int, default=8)
    p.add_argument("--fp_dim", type=int, default=128)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--T", type=int, default=2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=5)
    args = p.parse_args()

    device = mx.gpu if args.device == "gpu" else mx.cpu
    mx.set_default_device(device)

    model = AttentiveFPRegressor(
        args.n_atom, args.n_bond, args.fp_dim, args.radius, args.T,
        p_dropout=0.1, use_rms_norm=True,
    )
    edge_index, node_features, edge_features, batch_indices = make_batch(
        args.n_atom, args.n_bond, args.num_nodes, args.num_edges, args.batch_size,
    )

    for _ in range(args.warmup):
        run_forward_with_timing(
            model, edge_index, node_features, edge_features, batch_indices
        )

    all_times = []
    for _ in range(args.repeat):
        all_times.append(
            run_forward_with_timing(
                model, edge_index, node_features, edge_features, batch_indices
            )
        )

    # Aggregate: median per step
    step_names = sorted(all_times[0].keys())
    medians = {}
    for name in step_names:
        vals = [r[name] for r in all_times]
        vals.sort()
        medians[name] = vals[len(vals) // 2]

    total_ms = sum(medians.values())
    throughput = args.batch_size / (total_ms / 1000)

    print("=" * 72)
    print("AttentiveFP full speed / bottleneck analysis")
    print("=" * 72)
    print(f"  device={args.device}  batch_size={args.batch_size}")
    print(f"  num_nodes={args.num_nodes}  num_edges={args.num_edges}")
    print(f"  n_atom={args.n_atom}  n_bond={args.n_bond}  fp_dim={args.fp_dim}")
    print(f"  radius={args.radius}  T={args.T}")
    print(f"  Total (sum of steps): {total_ms:.3f} ms  ->  {throughput:.1f} graphs/s")
    print("  (Note: step-wise total > single end-to-end forward due to mx.eval() syncs;")
    print("   percentages show relative cost and what to optimize.)")
    print()
    print("  Step (bottlenecks first by % of total)")
    print("-" * 72)

    by_pct = [(name, medians[name], 100.0 * medians[name] / total_ms) for name in step_names]
    by_pct.sort(key=lambda x: -x[1])

    for name, ms, pct in by_pct:
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {name:<28}  {ms:>7.3f} ms  {pct:>5.1f}%  {bar}")

    print("-" * 72)
    print()
    print("  What to improve in MLX pipeline (by impact):")
    top = by_pct[:5]
    for i, (name, ms, pct) in enumerate(top, 1):
        print(f"    {i}. {name}: {pct:.1f}% — optimize this step first.")
    print()
    print("  Typical MLX optimizations:")
    print("    - scatter (align_scatter, attend_scatter): fused scatter-reduce or custom kernel")
    print("    - GRU: already using mx.fast.gru_cell on GPU; ensure inputs are contiguous")
    print("    - Linear / matmul: batch size and dims; check GEMM utilization")
    print("    - index / gather: node_features[src_idx], reduce redundant gathers")
    print("=" * 72)


if __name__ == "__main__":
    main()
