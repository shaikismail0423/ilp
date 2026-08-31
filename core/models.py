"""Core dataclasses for AutoAgentSearch.

Shared across task parser, component repository, search engines,
scoring / explanation modules and the visualisation layer.

An :class:`Architecture` may be represented in two forms:

* **Linear chain** — ``components`` is a flat list. This is what the beam
  and MCTS engines produce.
* **DAG (layered)** — ``layers`` is a list of *stages* where each stage
  contains one or more components that execute *in parallel*. Sequential
  layers wire in order; the last component of layer N feeds the first
  component of layer N+1 through an implicit fan-in.

The two views are kept consistent: when ``layers`` is set, ``components``
is auto-filled with the flattened view so any code that only understands
linear chains still works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Component:
    """A reusable AI building block described purely by metadata."""

    id: str
    name: str
    domain: str
    type: str  # input | retrieval | reasoning | validation | aggregator | output
    input: str
    output: str
    latency: float          # seconds
    cost: float             # USD per invocation
    memory: int             # MB
    quality: float          # 0..1
    cpu: bool               # whether it runs on CPU only

    @classmethod
    def from_dict(cls, d: dict) -> "Component":
        return cls(
            id=d["id"],
            name=d["name"],
            domain=d["domain"],
            type=d["type"],
            input=d["input"],
            output=d["output"],
            latency=float(d["latency"]),
            cost=float(d["cost"]),
            memory=int(d["memory"]),
            quality=float(d["quality"]),
            cpu=bool(d["cpu"]),
        )


@dataclass
class Task:
    """Structured representation of a user request."""

    domain: str
    intent: str
    input_type: str
    desired_output: str
    raw: str = ""


@dataclass
class Constraints:
    """Resource budgets provided by the user in the sidebar."""

    max_cost: float = 0.05           # USD
    max_latency: float = 3.0         # seconds
    max_memory: int = 4096           # MB
    cpu_only: bool = False


@dataclass
class Architecture:
    """A candidate workflow — linear chain OR DAG of layers."""

    components: List[Component] = field(default_factory=list)
    layers: Optional[List[List[Component]]] = None  # None ⇒ linear chain

    # cached aggregate metrics (populated by the scorer)
    total_cost: float = 0.0
    total_latency: float = 0.0
    peak_memory: int = 0
    aggregate_quality: float = 0.0
    score: float = 0.0

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_layers(cls, layers: List[List[Component]]) -> "Architecture":
        """Build an architecture from a list of parallel stages."""
        flat = [c for layer in layers for c in layer]
        return cls(components=flat, layers=layers)

    def __post_init__(self) -> None:
        # Keep ``components`` in sync with ``layers`` when the latter is
        # provided but the former is empty (or stale).
        if self.layers is not None and not self.components:
            self.components = [c for layer in self.layers for c in layer]

    # ------------------------------------------------------------ helpers
    @property
    def is_dag(self) -> bool:
        """True if any layer has more than one component (real parallelism)."""
        return self.layers is not None and any(len(layer) > 1 for layer in self.layers)

    @property
    def n_layers(self) -> int:
        return len(self.layers) if self.layers is not None else len(self.components)

    @property
    def max_width(self) -> int:
        if self.layers is None:
            return 1
        return max((len(layer) for layer in self.layers), default=0)

    def ids(self) -> List[str]:
        return [c.id for c in self.components]

    def signature(self) -> str:
        """Hashable identifier used for de-duplication in the beam."""
        if self.layers is None:
            return "->".join(self.ids())
        return " ~> ".join("+".join(c.id for c in layer) for layer in self.layers)

    def last_output(self) -> Optional[str]:
        if self.layers is not None and self.layers:
            return self.layers[-1][-1].output
        return self.components[-1].output if self.components else None
