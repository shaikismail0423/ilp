"""Hand-authored baseline architectures for the benchmark.

Following the PDF blueprint:

* **Fixed**          - a simple ``Planner -> Executor`` chain.
* **Industry**       - ``Planner -> Verifier -> Executor -> Report``,
  the pattern most enterprise agent frameworks ship out of the box.
* **Random**         - a randomly sampled valid chain of length 3-5.

These are what AutoAgentSearch's beam/MCTS results must beat to justify
the search.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from .models import Architecture, Component
from .repository import ComponentRepository
from .scorer import compute_metrics


# ---------------------------------------------------------------------- helpers
def _by_id(comps: List[Component]) -> Dict[str, Component]:
    return {c.id: c for c in comps}


def _make(comps: Dict[str, Component], ids: List[str]) -> Optional[Architecture]:
    """Build an :class:`Architecture` from a list of component ids, or
    ``None`` if any id is missing from the repository (shouldn't happen for
    the agentic domain)."""
    try:
        arch = Architecture(components=[comps[i] for i in ids])
    except KeyError:
        return None
    compute_metrics(arch)
    return arch


# ---------------------------------------------------------------------- baselines
def fixed_baseline(repo: ComponentRepository) -> Optional[Architecture]:
    """Planner -> Executor -> Report (unverified)."""
    comps = _by_id(repo.get("agentic"))
    return _make(comps, ["planner", "executor_raw", "status_report_raw"])


def industry_baseline(repo: ComponentRepository) -> Optional[Architecture]:
    """Planner -> Verifier -> Executor -> Report."""
    comps = _by_id(repo.get("agentic"))
    return _make(comps, ["planner", "verifier", "executor", "status_report_raw"])


def random_baseline(
    repo: ComponentRepository,
    length: int = 4,
    seed: int = 1337,
) -> Optional[Architecture]:
    """A random *type-valid* chain of the requested length.

    The chain must still respect input/output typing, so this is not a
    "random bag" - it is the fair worst case: any valid pipeline.
    """
    rng = random.Random(seed)
    pool = repo.get("agentic")
    inputs = [c for c in pool if c.type == "input"]
    if not inputs:
        return None

    chosen: List[Component] = [rng.choice(inputs)]
    used = {chosen[0].id}
    for _ in range(length - 1):
        last = chosen[-1]
        candidates = [
            c for c in pool
            if c.id not in used and c.input == last.output
        ]
        if not candidates:
            break
        pick = rng.choice(candidates)
        chosen.append(pick)
        used.add(pick.id)

    # Force termination at an output component if possible.
    if chosen[-1].type != "output":
        candidates = [c for c in pool
                      if c.type == "output" and c.input == chosen[-1].output
                      and c.id not in used]
        if candidates:
            chosen.append(rng.choice(candidates))

    arch = Architecture(components=chosen)
    compute_metrics(arch)
    return arch


def all_baselines(repo: ComponentRepository) -> Dict[str, Architecture]:
    """Return the three named baselines as a dict, skipping any that fail
    to build (defensive; on the shipped ``agentic.json`` all three build)."""
    out: Dict[str, Architecture] = {}
    for name, fn in [
        ("Fixed",     fixed_baseline),
        ("Industry",  industry_baseline),
        ("Random",    random_baseline),
    ]:
        arch = fn(repo)
        if arch is not None:
            out[name] = arch
    return out
