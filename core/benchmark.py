"""Three-engine benchmark: Beam Search vs MCTS vs Reinforcement Learning.

For every generated BPI-style task each engine is asked to produce one
architecture. Every architecture is *guaranteed valid* by the engine:
type-compatible, cycle-free, in-budget, and terminating at an output node.

Each architecture is then simulated ``repeats`` times per task with
Monte-Carlo variance, and again ``repeats`` times per task per injected
failure mode for the Recovery Rate metric.

Statistical significance between every pair of engines is reported with
paired t-test and Wilcoxon signed-rank tests (SciPy-preferred with a
pure-Python fallback).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .beam_search import beam_search
from .bpi_tasks import BPITask, generate_tasks
from .mcts_search import mcts_search
from .models import Architecture, Constraints, Task
from .repository import ComponentRepository
from .rl_search import rl_search
from .simulator import simulate_pipeline


# ---------------------------------------------------------------------- config
_INJECTED_FAILURES = ("missing_activity", "incorrect_dependency", "resource_shortage")


@dataclass
class EngineScores:
    """Per-engine metrics averaged over ``repeats`` runs across ``n_tasks``."""
    name: str
    success_rate: float
    resource_efficiency: float
    latency: float
    memory: int
    complexity: int
    recovery_rate: float
    per_task_success: List[float] = field(default_factory=list)
    # A representative winning architecture (from the first task) for display
    example_pipeline: List[str] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    scores: Dict[str, EngineScores]
    pairwise_stats: Dict[str, Dict[str, float]]   # {"Beam vs MCTS": {...}}
    n_tasks: int
    n_repeats: int


# ---------------------------------------------------------------------- helpers
def _bpi_to_task(bpi: BPITask) -> Task:
    return Task(
        domain="agentic",
        intent="process_execution",
        input_type="task",
        desired_output="report",
        raw=bpi.as_task_text(),
    )


def _run_arch(
    arch: Architecture, bpi: BPITask, repeats: int, base_seed: int,
) -> Dict[str, float]:
    successes = 0
    latencies: List[float] = []
    costs: List[float] = []
    peak_mems: List[int] = []
    for r in range(repeats):
        res = simulate_pipeline(arch, bpi, seed=base_seed + r)
        successes += int(res.success)
        latencies.append(res.latency)
        costs.append(res.cost)
        peak_mems.append(res.peak_memory)
    return {
        "success_rate": successes / repeats,
        "latency": statistics.mean(latencies),
        "cost": statistics.mean(costs),
        "memory": max(peak_mems),
    }


def _run_robustness(
    arch: Architecture, bpi: BPITask, repeats: int, base_seed: int,
) -> float:
    if not arch.components:
        return 0.0
    total, hits = 0, 0
    for i, mode in enumerate(_INJECTED_FAILURES):
        for r in range(repeats):
            res = simulate_pipeline(
                arch, bpi,
                seed=base_seed + 1000 * (i + 1) + r,
                inject_failure=mode,
            )
            total += 1
            hits += int(res.success)
    return hits / max(total, 1)


# ---------------------------------------------------------------------- stats
def _paired_t_p(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) != len(b) or len(a) < 2:
        return None
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diffs):
        return 1.0
    try:
        from scipy import stats  # type: ignore
        return float(stats.ttest_rel(a, b).pvalue)
    except Exception:
        n = len(diffs)
        mean = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        if sd == 0:
            return 1.0
        t = mean / (sd / math.sqrt(n))
        z = abs(t)
        p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
        return float(p)


def _wilcoxon_p(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) != len(b) or len(a) < 2:
        return None
    try:
        from scipy import stats  # type: ignore
        diffs = [x - y for x, y in zip(a, b)]
        if all(d == 0 for d in diffs):
            return 1.0
        return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception:
        diffs = [x - y for x, y in zip(a, b) if x != y]
        if not diffs:
            return 1.0
        abs_diffs = sorted(enumerate(diffs), key=lambda t: abs(t[1]))
        ranks = [0.0] * len(abs_diffs)
        i = 0
        while i < len(abs_diffs):
            j = i
            while j + 1 < len(abs_diffs) and abs(abs_diffs[j + 1][1]) == abs(abs_diffs[i][1]):
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[k] = avg_rank
            i = j + 1
        w_plus = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d > 0)
        n = len(diffs)
        mu = n * (n + 1) / 4
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        if sigma == 0:
            return 1.0
        z = (w_plus - mu) / sigma
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return float(p)


# ---------------------------------------------------------------------- driver
def run_benchmark(
    repo: ComponentRepository,
    cons: Constraints,
    n_tasks: int = 20,
    repeats: int = 10,
    beam_width: int = 5,
    mcts_iterations: int = 400,
    rl_episodes: int = 300,
    seed: int = 42,
) -> BenchmarkReport:
    """Run every engine on every task, then compute pairwise statistics."""
    tasks_bpi = generate_tasks(n=n_tasks, seed=seed)

    engines: Dict[str, Callable[[BPITask], Optional[Architecture]]] = {
        "Beam Search": lambda bpi: (
            beam_search(repo, _bpi_to_task(bpi), cons,
                        beam_width=beam_width, top_k=1)[:1] or [None]
        )[0],
        "MCTS": lambda bpi: (
            mcts_search(repo, _bpi_to_task(bpi), cons,
                        iterations=mcts_iterations, top_k=1)[:1] or [None]
        )[0],
        "Reinforcement Learning": lambda bpi: (
            rl_search(repo, _bpi_to_task(bpi), cons,
                      episodes=rl_episodes, top_k=1)[:1] or [None]
        )[0],
    }

    per_engine_success: Dict[str, List[float]] = {}
    aggregate: Dict[str, EngineScores] = {}

    for name, builder in engines.items():
        per_task: List[Dict[str, float]] = []
        per_task_robust: List[float] = []
        complexities: List[int] = []
        example_pipeline: List[str] = []

        for i, bpi in enumerate(tasks_bpi):
            arch = builder(bpi)
            if arch is None or not arch.components:
                per_task.append({"success_rate": 0.0, "latency": 0.0,
                                 "cost": 0.0, "memory": 0})
                per_task_robust.append(0.0)
                complexities.append(0)
                continue
            per_task.append(_run_arch(arch, bpi, repeats, seed))
            per_task_robust.append(_run_robustness(arch, bpi, repeats, seed))
            complexities.append(len(arch.components))
            if i == 0:
                example_pipeline = [c.name for c in arch.components]

        succ = [x["success_rate"] for x in per_task]
        lat = statistics.mean(x["latency"] for x in per_task)
        cost = statistics.mean(x["cost"] for x in per_task)
        mem = int(max(x["memory"] for x in per_task))
        sr = statistics.mean(succ)
        aggregate[name] = EngineScores(
            name=name,
            success_rate=sr,
            resource_efficiency=sr / max(cost, 1e-6),
            latency=lat,
            memory=mem,
            complexity=int(round(statistics.mean(complexities))) if complexities else 0,
            recovery_rate=statistics.mean(per_task_robust),
            per_task_success=succ,
            example_pipeline=example_pipeline,
        )
        per_engine_success[name] = succ

    # Pairwise statistics
    pairs = [("Beam Search", "MCTS"),
             ("Beam Search", "Reinforcement Learning"),
             ("MCTS", "Reinforcement Learning")]
    pairwise_stats: Dict[str, Dict[str, float]] = {}
    for a, b in pairs:
        A = per_engine_success.get(a, [])
        B = per_engine_success.get(b, [])
        pairwise_stats[f"{a} vs {b}"] = {
            "mean_delta": (statistics.mean(A) - statistics.mean(B))
                          if A and B else float("nan"),
            "t_p": _paired_t_p(A, B) or float("nan"),
            "wilcoxon_p": _wilcoxon_p(A, B) or float("nan"),
        }

    return BenchmarkReport(
        scores=aggregate,
        pairwise_stats=pairwise_stats,
        n_tasks=n_tasks,
        n_repeats=repeats,
    )
