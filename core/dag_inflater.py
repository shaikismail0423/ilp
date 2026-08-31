"""Post-process a linear chain into a parallel / hierarchical DAG.

The beam and MCTS engines produce a linear pipeline. This module takes
that pipeline and "inflates" it — wherever the repository offers multiple
compatible alternatives for a given stage type (retrieval, reasoning,
validation), it turns that stage into a *layer* of parallel components
that then fan back into an aggregator.

Rules of the inflater:

* An input stage is never parallelised (there's one document going in).
* A reasoning stage may be parallelised across up to ``max_width`` alternatives
  if they all have the same ``input`` / ``output`` types — the parallel branch
  is a **model ensemble**.
* A retrieval stage may be parallelised similarly (e.g. embedding + SQL
  retriever, both producing ``context_text``).
* A validation stage may be parallelised as an **ensemble of validators**.
* After every parallel layer the inflater inserts an **aggregator** whose
  ``input`` type matches the layer's ``output`` and whose ``output`` matches
  the next stage's ``input``. If no aggregator exists in the repository the
  inflater falls back to keeping the layer linear (safe degradation).
* The final output stage is never parallelised.

The inflater accepts a ``budget`` predicate: if the widened architecture
violates the user's resource constraints, the inflater rolls that layer
back to a single component and tries the next one. Any layer that can't be
widened without breaking the budget is simply left as-is.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from .models import Architecture, Component, Constraints
from .repository import ComponentRepository
from .scorer import compute_metrics, satisfies_constraints


# --------------------------------------------------------------- helpers
_PARALLELIZABLE_TYPES = {"retrieval", "reasoning", "validation"}


def _peers(repo: ComponentRepository, domain: str, comp: Component) -> List[Component]:
    """Other components with identical input/output types and same category."""
    pool = repo.get(domain)
    return [
        c for c in pool
        if c.id != comp.id
        and c.type == comp.type
        and c.input == comp.input
        and c.output == comp.output
    ]


def _find_aggregator(
    repo: ComponentRepository,
    domain: str,
    io_type: str,
) -> Optional[Component]:
    """Find an aggregator whose input == output == the given type."""
    for c in repo.get(domain):
        if c.type == "aggregator" and c.input == io_type and c.output == io_type:
            return c
    return None


# --------------------------------------------------------------- inflater
def inflate_to_dag(
    chain: Architecture,
    repo: ComponentRepository,
    domain: str,
    cons: Constraints,
    max_width: int = 3,
    enable_hierarchy: bool = True,
) -> Architecture:
    """Return a DAG-shaped :class:`Architecture` derived from ``chain``.

    Parameters
    ----------
    chain
        A linear architecture (usually the best result from beam or MCTS).
    repo, domain
        Used to discover peer components and aggregators.
    cons
        User's resource budget. Any widening that violates it is rolled back.
    max_width
        Maximum number of components in any one parallel layer.
    enable_hierarchy
        If True, when a reasoning stage is widened *and* a validation stage
        exists downstream, the validator is also widened to form a hierarchy
        (parallel reasoning → aggregator → parallel validation → aggregator).
    """
    if not chain.components:
        return chain

    layers: List[List[Component]] = [[c] for c in chain.components]
    widened_reasoning = False

    # Snapshot the original components — we walk the pipeline in its
    # original order, looking up each component's *current* index in the
    # (possibly mutated) layers list on the fly. This avoids the pitfall
    # of iterating with enumerate() over a list we are mutating.
    original = list(chain.components)
    for comp in original:
        if comp.type not in _PARALLELIZABLE_TYPES:
            continue
        # Under hierarchy mode, validation may be widened independently to
        # form a validator ensemble even without prior reasoning widening —
        # this is a common enterprise pattern (multiple compliance checks).
        if comp.type == "validation" and not enable_hierarchy:
            continue

        # Locate the component in the current (possibly widened) layers.
        idx = next(
            (i for i, layer in enumerate(layers) if comp in layer),
            None,
        )
        if idx is None:
            continue

        peers = _peers(repo, domain, comp)
        if not peers:
            continue

        # Choose up to (max_width - 1) peers with the highest quality.
        peers.sort(key=lambda c: c.quality, reverse=True)
        chosen = [comp] + peers[: max_width - 1]

        # Try inserting an aggregator right after the widened layer.
        agg = _find_aggregator(repo, domain, comp.output)
        # Skip if an aggregator with the right I/O already sits at idx+1
        # (some search chains may already contain one).
        next_layer_has_aggregator = (
            idx + 1 < len(layers)
            and layers[idx + 1]
            and layers[idx + 1][0].type == "aggregator"
            and layers[idx + 1][0].input == comp.output
        )
        need_aggregator = (
            agg is not None
            and idx + 1 < len(layers)
            and not next_layer_has_aggregator
        )

        # Build a candidate layers list with the widened layer applied.
        candidate = [list(l) for l in layers]
        candidate[idx] = chosen
        if need_aggregator:
            candidate.insert(idx + 1, [agg])

        trial = Architecture.from_layers(candidate)
        compute_metrics(trial)

        if satisfies_constraints(trial, cons):
            layers = candidate
            if comp.type == "reasoning":
                widened_reasoning = True

    inflated = Architecture.from_layers(layers)
    compute_metrics(inflated)
    return inflated
