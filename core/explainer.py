"""Natural-language explanations for the search results.

The explanations do not rely on an LLM - they are template-driven, but the
templates read the actual metrics of each architecture so the output feels
tailored rather than generic.
"""

from __future__ import annotations

from typing import List

from .models import Architecture, Constraints, Task


def _fmt_money(x: float) -> str:
    return f"${x:.4f}"


def _fmt_time(x: float) -> str:
    return f"{x:.2f} s"


def _fmt_mem(x: int) -> str:
    return f"{x / 1024:.2f} GB" if x >= 1024 else f"{x} MB"


def explain_best(arch: Architecture, task: Task, cons: Constraints) -> str:
    """One-paragraph justification for the winning architecture."""
    stages = " -> ".join(c.name for c in arch.components)
    reasons: List[str] = []

    if arch.total_cost <= cons.max_cost:
        reasons.append(
            f"stays within your budget of {_fmt_money(cons.max_cost)} "
            f"(uses {_fmt_money(arch.total_cost)})"
        )
    if arch.total_latency <= cons.max_latency:
        reasons.append(
            f"meets the latency ceiling of {_fmt_time(cons.max_latency)} "
            f"(runs in {_fmt_time(arch.total_latency)})"
        )
    if arch.peak_memory <= cons.max_memory:
        reasons.append(
            f"fits inside {_fmt_mem(cons.max_memory)} of memory "
            f"(peaks at {_fmt_mem(arch.peak_memory)})"
        )
    if cons.cpu_only:
        reasons.append("runs entirely on CPU")

    quality_pct = arch.aggregate_quality * 100
    reasons.append(f"achieves the highest predicted quality of {quality_pct:.1f}%")

    reason_txt = ", ".join(reasons[:-1]) + (
        f", and {reasons[-1]}" if len(reasons) > 1 else reasons[0]
    )
    return (
        f"For the {task.domain} task '{task.intent.replace('_', ' ')}', "
        f"the pipeline {stages} was selected because it {reason_txt} "
        f"among all feasible workflows the search explored."
    )


def explain_alternative(best: Architecture, alt: Architecture) -> str:
    """Why an alternative ranked lower than the winner."""
    reasons: List[str] = []
    if alt.total_cost > best.total_cost:
        reasons.append(f"costs more ({_fmt_money(alt.total_cost)} vs {_fmt_money(best.total_cost)})")
    elif alt.total_cost < best.total_cost:
        reasons.append(f"is cheaper ({_fmt_money(alt.total_cost)}) but sacrifices quality")

    if alt.total_latency > best.total_latency:
        reasons.append(f"is slower ({_fmt_time(alt.total_latency)} vs {_fmt_time(best.total_latency)})")
    elif alt.total_latency < best.total_latency:
        reasons.append(f"is faster ({_fmt_time(alt.total_latency)}) but weaker in reasoning quality")

    if alt.peak_memory > best.peak_memory:
        reasons.append(f"needs more memory ({_fmt_mem(alt.peak_memory)} vs {_fmt_mem(best.peak_memory)})")
    elif alt.peak_memory < best.peak_memory:
        reasons.append(f"is lighter on memory ({_fmt_mem(alt.peak_memory)}) but drops accuracy")

    q_diff = (best.aggregate_quality - alt.aggregate_quality) * 100
    if q_diff > 0.1:
        reasons.append(f"has {q_diff:.1f}% lower predicted quality")

    if not reasons:
        reasons.append("scored marginally lower on the combined metric")
    return "Ranked lower because it " + "; ".join(reasons) + "."
