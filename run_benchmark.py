"""Command-line entry point for the three-engine research benchmark.

Compares Beam Search vs Monte-Carlo Tree Search vs Reinforcement Learning
across a set of synthetic BPI-style tasks. Every engine returns a valid
architecture; each is simulated multiple times per task with injected
failures for the Recovery Rate metric.

Run with:
    python run_benchmark.py
    python run_benchmark.py --tasks 30 --repeats 15 --rl 500
"""

from __future__ import annotations

import argparse
import os
import sys

from core.benchmark import run_benchmark
from core.models import Constraints
from core.repository import ComponentRepository


def _fmt_p(p: float) -> str:
    if p != p:          # NaN
        return "  n/a "
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Beam vs MCTS vs RL benchmark")
    ap.add_argument("--tasks",   type=int,   default=20)
    ap.add_argument("--repeats", type=int,   default=10)
    ap.add_argument("--beam",    type=int,   default=5)
    ap.add_argument("--mcts",    type=int,   default=400)
    ap.add_argument("--rl",      type=int,   default=300,
                    help="RL training episodes per task")
    ap.add_argument("--budget",  type=float, default=0.05)
    ap.add_argument("--latency", type=float, default=5.0)
    ap.add_argument("--memory",  type=int,   default=8192)
    args = ap.parse_args()

    repo = ComponentRepository(
        os.path.join(os.path.dirname(__file__), "components")
    )
    cons = Constraints(
        max_cost=args.budget,
        max_latency=args.latency,
        max_memory=args.memory,
    )

    print(f"Running benchmark: {args.tasks} tasks x {args.repeats} repeats  "
          f"(beam={args.beam}, mcts={args.mcts}, rl={args.rl})")
    report = run_benchmark(
        repo, cons,
        n_tasks=args.tasks,
        repeats=args.repeats,
        beam_width=args.beam,
        mcts_iterations=args.mcts,
        rl_episodes=args.rl,
    )

    print("\n" + "=" * 82)
    print(f"{'Search Engine':<28} {'Success':>8} {'Resource':>10} {'Latency':>9} "
          f"{'Memory':>8} {'#Ag':>4} {'Recovery':>9}")
    print(f"{'':<28} {'Rate':>8} {'Efficiency':>10} {'':>9} "
          f"{'':>8} {'':>4} {'Rate':>9}")
    print("-" * 82)
    for name, s in report.scores.items():
        print(f"{name:<28} {s.success_rate*100:>7.1f}% "
              f"{s.resource_efficiency:>10.1f} "
              f"{s.latency:>7.2f}s "
              f"{s.memory:>6} MB "
              f"{s.complexity:>4} "
              f"{s.recovery_rate*100:>8.1f}%")

    print("\nExample architecture on task #0:")
    for name, s in report.scores.items():
        print(f"  {name:<28}  " + " -> ".join(s.example_pipeline))

    print("\nPairwise statistical comparison:")
    print("-" * 82)
    print(f"{'Comparison':<44} {'Δ success':>12} {'t-test p':>10} {'Wilcoxon p':>12}")
    for pair, st in report.pairwise_stats.items():
        print(f"{pair:<44} {st['mean_delta']*100:>+11.1f}% "
              f"{_fmt_p(st['t_p']):>10} {_fmt_p(st['wilcoxon_p']):>12}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
