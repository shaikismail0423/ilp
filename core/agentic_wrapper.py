"""Wrap a domain-specific pipeline in an agentic orchestration scaffold.

The agentic roles used by the wrapper (Planner, Verifier, Recovery Agent,
Status Report) are re-typed with **universal I/O** (``"*"``) so the
wrapped architecture is type-consistent end-to-end regardless of what
concrete types flow through the domain body. This mirrors how these
agents actually work in a real system — they don't transform the data,
they observe or command the pipeline around it.

Result: a fully **valid** architecture — every edge is
type-compatible, no redundant no-op stages, ends at a proper output.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import Architecture, Component, Constraints
from .repository import ComponentRepository
from .scorer import compute_metrics, satisfies_constraints


# The four orchestration roles the wrapper looks for in the agentic library.
_ROLE_TO_ID = {
    "planner":  "planner",
    "verifier": "verifier",
    "recovery": "recovery_agent",
    "report":   "status_report",
}

_UNIVERSAL = "*"


def _base(repo: ComponentRepository, comp_id: str) -> Optional[Component]:
    for c in repo.get("agentic"):
        if c.id == comp_id:
            return c
    return None


def _universalise(base: Component, output: str = _UNIVERSAL) -> Component:
    """Return a copy of ``base`` re-typed with universal ``input`` and the
    given ``output`` (``"*"`` by default). Metrics are preserved."""
    return Component(
        id=base.id + "_orch",
        name=base.name,
        domain=base.domain,
        type=base.type,
        input=_UNIVERSAL,
        output=output,
        latency=base.latency,
        cost=base.cost,
        memory=base.memory,
        quality=base.quality,
        cpu=base.cpu,
    )


def wrap_in_agentic_scaffold(
    arch: Architecture,
    repo: ComponentRepository,
    cons: Optional[Constraints] = None,
    keep_domain_output: bool = True,
) -> Architecture:
    """Return an agentic-orchestrated version of the domain pipeline.

    The wrapping is applied only when it stays within the budget; otherwise
    the original pipeline is returned unchanged (safe degradation).
    """
    if not arch.components:
        return arch

    p = _base(repo, _ROLE_TO_ID["planner"])
    v = _base(repo, _ROLE_TO_ID["verifier"])
    r = _base(repo, _ROLE_TO_ID["recovery"])
    s = _base(repo, _ROLE_TO_ID["report"])
    if p is None or v is None or r is None:
        return arch

    # Universalise the orchestrators so the wrapped chain type-checks.
    planner  = _universalise(p)
    verifier = _universalise(v)
    recovery = _universalise(r)
    # Status Report replaces the domain output when keep_domain_output=False.
    status   = _universalise(s, output="report") if s is not None else None

    # Split off the trailing domain-output stage (if any) so we can decide
    # whether to keep it or replace it with the agentic Status Report.
    if arch.layers is not None:
        base_layers: List[List[Component]] = [list(l) for l in arch.layers]
    else:
        base_layers = [[c] for c in arch.components]

    domain_output_layer: Optional[List[Component]] = None
    if base_layers and base_layers[-1] and base_layers[-1][0].type == "output":
        domain_output_layer = base_layers.pop()

    wrapped: List[List[Component]] = []
    wrapped.append([planner])          # ORCHESTRATION HEAD
    wrapped.append([verifier])         # plan-level check
    wrapped.extend(base_layers)        # ---- domain body (unchanged) ----
    wrapped.append([recovery])         # error handling & retry
    if keep_domain_output and domain_output_layer is not None:
        wrapped.append(domain_output_layer)
    elif status is not None:
        wrapped.append([status])
    elif domain_output_layer is not None:
        wrapped.append(domain_output_layer)

    scaffolded = Architecture.from_layers(wrapped)
    compute_metrics(scaffolded)

    if cons is not None and not satisfies_constraints(scaffolded, cons):
        return arch

    return scaffolded


# --------------------------------------------------------------------------- validity
def is_valid_architecture(arch: Architecture) -> bool:
    """Full validity check across a linear or layered architecture.

    * Every edge is type-compatible (with ``"*"`` universal).
    * No component id appears twice (no cycles).
    * Terminates at an ``output`` stage.
    """
    from .type_check import matches
    if not arch.components:
        return False

    # Cycle / duplicate check on ids.
    ids = arch.ids()
    if len(set(ids)) != len(ids):
        return False

    # Terminal check.
    tail = arch.components[-1]
    if tail.type != "output":
        return False

    # Type-flow check across successive layers (or successive components).
    if arch.layers is not None:
        for i in range(len(arch.layers) - 1):
            prev_layer = arch.layers[i]
            next_layer = arch.layers[i + 1]
            for nxt in next_layer:
                # Every downstream component must be reachable from at least
                # one upstream component in the immediate previous layer.
                if not any(matches(p.output, nxt.input) for p in prev_layer):
                    return False
    else:
        for i in range(len(arch.components) - 1):
            if not matches(arch.components[i].output, arch.components[i + 1].input):
                return False
    return True
