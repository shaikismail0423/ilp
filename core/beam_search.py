"""Beam-search architecture engine.

This is the heart of AutoAgentSearch. Given a parsed :class:`Task` and a
:class:`Constraints` budget, the engine explores the space of possible
workflows built from the domain's component repository and returns the top-K
architectures ranked by :func:`scorer.compute_metrics`.

The search is structured as a classical beam search where each state is a
partial pipeline (an :class:`Architecture` chain). At every expansion step,
every state in the beam is extended by every compatible next component, all
successors are scored, invalid or over-budget candidates are pruned, and the
top ``beam_width`` survivors move on to the next depth.
"""

from __future__ import annotations

from typing import Dict, List, Set

from .models import Architecture, Component, Constraints, Task
from .repository import ComponentRepository
from .scorer import compute_metrics, satisfies_constraints, rank


# The pipelines we care about are short (5-8 stages). A hard depth cap keeps
# the search finite and stops the beam from producing degenerate loops.
_MAX_DEPTH = 15
_TERMINAL_OUTPUTS = {"json", "report"}


# --------------------------------------------------------------------------- helpers
def _valid_starts(repo: ComponentRepository, task: Task) -> List[Component]:
    """Input components whose input type matches the task's input type."""
    inputs = repo.by_type(task.domain, "input")
    return [c for c in inputs if c.input == task.input_type] or inputs


def _successors(
    repo: ComponentRepository,
    arch: Architecture,
    task: Task,
    cons: Constraints,
) -> List[Component]:
    """Return the components that may be appended to ``arch``."""
    last = arch.components[-1]
    used: Set[str] = {c.id for c in arch.components}

    candidates: List[Component] = []
    for c in repo.get(task.domain):
        if c.id in used:                # no cycles / no repeats
            continue
        if c.input != last.output:      # type compatibility
            continue
        if cons.cpu_only and not c.cpu: # honour the CPU-only switch early
            continue
        # Parsimony rule: never chain two "input" components in a row —
        # such a chain is always a semantic no-op.
        if last.type == "input" and c.type == "input":
            continue
        candidates.append(c)
    return candidates


def _is_terminal(arch: Architecture, task: Task) -> bool:
    """A pipeline is complete when it ends at an output component whose
    output matches (or is compatible with) the user's desired output."""
    if not arch.components:
        return False
    tail = arch.components[-1]
    if tail.type != "output":
        return False
    if tail.output not in _TERMINAL_OUTPUTS:
        return False
    # The desired output is a preference, not a hard rule - a report is still
    # useful when the user asked for json and vice versa, so we accept both
    # but rank exact matches higher via the quality metric.
    return True


# --------------------------------------------------------------------------- search
def beam_search(
    repo: ComponentRepository,
    task: Task,
    cons: Constraints,
    beam_width: int = 5,
    top_k: int = 3,
) -> List[Architecture]:
    """Run beam search and return up to ``top_k`` best architectures.

    Parameters
    ----------
    repo : ComponentRepository
        The pool of components (already loaded for the task's domain).
    task : Task
        The parsed user request.
    cons : Constraints
        Sidebar-provided resource budgets.
    beam_width : int, default 5
        Number of partial states kept between expansion rounds.
    top_k : int, default 3
        Number of completed pipelines returned at the end.
    """
    # Seed the beam with every valid starting component.
    beam: List[Architecture] = []
    for start in _valid_starts(repo, task):
        arch = Architecture(components=[start])
        compute_metrics(arch)
        beam.append(arch)
    beam = rank(beam)[:beam_width]

    completed: Dict[str, Architecture] = {}

    for _depth in range(_MAX_DEPTH):
        expanded: List[Architecture] = []
        for arch in beam:
            # Any state that is already a valid, in-budget terminal is
            # recorded before we try to extend it further.
            if _is_terminal(arch, task) and satisfies_constraints(arch, cons):
                completed[arch.signature()] = arch

            for nxt in _successors(repo, arch, task, cons):
                new = Architecture(components=arch.components + [nxt])
                compute_metrics(new)

                # Prune anything that has already blown the budget - a
                # longer pipeline can only make cost/latency worse.
                if new.total_cost > cons.max_cost:
                    continue
                if new.total_latency > cons.max_latency:
                    continue
                if new.peak_memory > cons.max_memory:
                    continue
                expanded.append(new)

        if not expanded:
            break
        # De-duplicate then keep the best `beam_width` for the next round.
        seen: Set[str] = set()
        deduped: List[Architecture] = []
        for a in rank(expanded):
            if a.signature() in seen:
                continue
            seen.add(a.signature())
            deduped.append(a)
        beam = deduped[:beam_width]

    # Sweep the final beam for any remaining terminals.
    for arch in beam:
        if _is_terminal(arch, task) and satisfies_constraints(arch, cons):
            completed.setdefault(arch.signature(), arch)

    return rank(list(completed.values()))[:top_k]
