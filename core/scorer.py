"""Scoring engine for candidate architectures.

Supports two shapes of architecture:

* **Linear chain** — the historical case. Metrics aggregate additively over
  the ordered sequence.
* **Layered DAG** — layers execute in parallel; sequential layers execute in
  order. Latency becomes the critical path (sum of the slowest node per
  layer), memory becomes the peak concurrent footprint (sum of memory
  within a layer, then max across layers). Cost still totals every
  invocation. Quality still uses the joint probability that every stage in
  the DAG works.
"""

from __future__ import annotations

from functools import reduce
from typing import List

from .models import Architecture, Constraints


# Normalisation constants — the "worst reasonable" value for each metric.
# Chosen to match the UI's max sliders so the score keeps discriminating
# right up to the top of the range (before the fix these were 0.02, 4.0
# and 4096 and the score saturated in the upper half of the sliders).
_COST_NORM    = 0.10    # $
_LATENCY_NORM = 10.0    # seconds
_MEMORY_NORM  = 16384   # MB

_W_QUALITY = 0.55
_W_COST = 0.20
_W_LATENCY = 0.15
_W_MEMORY = 0.10


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def compute_metrics(arch: Architecture) -> Architecture:
    """Populate the aggregate metric fields on the architecture in-place."""
    if not arch.components:
        return arch

    # Cost is always the total number of invocations — a parallel branch
    # still costs compute, it just happens at the same wall-clock time.
    arch.total_cost = sum(c.cost for c in arch.components)

    if arch.layers is None:
        # Linear-chain semantics (unchanged from v1).
        arch.total_latency = sum(c.latency for c in arch.components)
        arch.peak_memory = max(c.memory for c in arch.components)
    else:
        # DAG semantics.
        # Latency = critical path = sum over layers of the slowest node in
        # that layer (parallel nodes overlap in time within a layer).
        arch.total_latency = sum(
            max(c.latency for c in layer) if layer else 0.0
            for layer in arch.layers
        )
        # Memory = peak concurrent footprint = max over layers of the sum
        # of memory of nodes within the layer (parallel nodes run at the
        # same time so their memory adds up while that layer is active).
        arch.peak_memory = max(
            (sum(c.memory for c in layer) for layer in arch.layers if layer),
            default=0,
        )

    # Quality is treated as the joint probability that every stage does its
    # job correctly - this rewards short, high-quality pipelines and punishes
    # unnecessary weak links.
    arch.aggregate_quality = reduce(lambda acc, c: acc * c.quality, arch.components, 1.0)

    q = arch.aggregate_quality
    c_norm = _clip(arch.total_cost / _COST_NORM)
    l_norm = _clip(arch.total_latency / _LATENCY_NORM)
    m_norm = _clip(arch.peak_memory / _MEMORY_NORM)

    arch.score = (
        _W_QUALITY * q
        - _W_COST * c_norm
        - _W_LATENCY * l_norm
        - _W_MEMORY * m_norm
    )
    return arch


def satisfies_constraints(arch: Architecture, cons: Constraints) -> bool:
    """Return True iff every constraint is respected by this architecture."""
    if arch.total_cost > cons.max_cost:
        return False
    if arch.total_latency > cons.max_latency:
        return False
    if arch.peak_memory > cons.max_memory:
        return False
    if cons.cpu_only and any(not c.cpu for c in arch.components):
        return False
    return True


def rank(archs: List[Architecture]) -> List[Architecture]:
    """Return archs sorted from best (highest score) to worst.

    Precondition: every architecture has already been scored (i.e.
    :func:`compute_metrics` was called on it). Search engines and the DAG
    inflater always score before ranking; if you rank a batch you built by
    hand, call :func:`compute_metrics` on each entry first.
    """
    return sorted(archs, key=lambda a: a.score, reverse=True)
