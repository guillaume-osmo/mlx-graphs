# Copyright © 2023-2024 Apple Inc.
# Compare SmilesX-style sequence GRU speed: legacy (Python) vs fast (Metal).
# SmilesX uses RNN (LSTM/GRU) on SMILES sequences; this benchmarks nn.GRU
# with MLX_RNN_IMPL=legacy vs fast in a sequence-to-vector setup.
#
# Run from repo root: python benchmarks/compare_smilesx_style_legacy_fast.py [--device gpu]

import argparse
import os
import subprocess
import sys
import time


def run_one(impl: str, device: str, warmup: int, repeat: int, **kwargs) -> float:
    """Run SmilesX-style GRU benchmark in a subprocess with MLX_RNN_IMPL=impl."""
    code = f"""
import os
import time
os.environ["MLX_RNN_IMPL"] = "{impl}"
import mlx.core as mx
import mlx.nn as nn

batch = {kwargs.get('batch_size', 32)}
seq_len = {kwargs.get('seq_len', 80)}
input_size = {kwargs.get('input_size', 64)}
hidden_size = {kwargs.get('hidden_size', 128)}
warmup = {warmup}
repeat = {repeat}

mx.set_default_device(mx.gpu if "{device}" == "gpu" else mx.cpu)
gru = nn.GRU(input_size, hidden_size, bias=True)
x = mx.random.normal((batch, seq_len, input_size)).astype(mx.float32)
h0 = mx.zeros((batch, hidden_size)).astype(mx.float32)

def forward():
    out = gru(x, h0)
    mx.eval(out)
    return out

for _ in range(warmup):
    forward()

times_ms = []
for _ in range(repeat):
    t0 = time.perf_counter()
    forward()
    t1 = time.perf_counter()
    times_ms.append((t1 - t0) * 1000)
times_ms.sort()
median_ms = times_ms[len(times_ms) // 2]
print(median_ms)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "MLX_RNN_IMPL": impl},
        capture_output=True,
        text=True,
        timeout=120,
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
    )
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        return float("nan")
    try:
        return float(result.stdout.strip())
    except ValueError:
        return float("nan")


def main():
    p = argparse.ArgumentParser(
        description="Compare SmilesX-style GRU: legacy vs fast (nn.GRU sequence)"
    )
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=80)
    p.add_argument("--input_size", type=int, default=64)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=5)
    args = p.parse_args()

    kwargs = {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "input_size": args.input_size,
        "hidden_size": args.hidden_size,
    }

    print("SmilesX-style sequence GRU: legacy vs fast (nn.GRU)")
    print(f"  device={args.device}  batch={args.batch_size}  seq_len={args.seq_len}")
    print(f"  input_size={args.input_size}  hidden_size={args.hidden_size}")
    print(f"  median of {args.repeat} runs (warmup={args.warmup})")
    print()

    legacy_ms = run_one("legacy", args.device, args.warmup, args.repeat, **kwargs)
    fast_ms = run_one("fast", args.device, args.warmup, args.repeat, **kwargs)

    print("=" * 60)
    print("  Mode              Median (ms)")
    print("=" * 60)
    print(f"  legacy GRU (Python)   {legacy_ms:.3f}")
    print(f"  fast GRU (Metal)      {fast_ms:.3f}")
    print("=" * 60)
    if fast_ms > 0 and legacy_ms == legacy_ms:
        print(f"  Speedup (fast vs legacy): {legacy_ms / fast_ms:.2f}x")
    print()
    print("  Uses MLX nn.GRU; MLX_RNN_IMPL=legacy|fast controls the path.")


if __name__ == "__main__":
    main()
