"""Type-compatibility rules shared by every search engine and the wrapper.

Two components ``a`` and ``b`` can be connected as ``a -> b`` if
``matches(a.output, b.input)`` is True.

The special string ``"*"`` acts as a universal type — it matches any
concrete type. This is used by the four orchestration components
(Planner, Verifier, Recovery Agent, Status Report) which do not transform
data; they wrap the pipeline as agents that observe or command it and are
compatible with whatever concrete type flows through them.

A second rule enforced here is *parsimony*: the search must not chain two
"input" components in a row, because such a chain is always a semantic
no-op (an ingestor that takes what a parser already produced, for
example). This is what stops "PDF Parser -> Plain Text Ingestor -> …"
pipelines from being returned.
"""

from __future__ import annotations

from typing import List

from .models import Component


_UNIVERSAL = "*"


def matches(produced: str, expected: str) -> bool:
    """Return True if a stage producing ``produced`` can feed one expecting
    ``expected``. ``"*"`` on either side is universal."""
    return produced == expected or produced == _UNIVERSAL or expected == _UNIVERSAL


def can_append(chain: List[Component], candidate: Component) -> bool:
    """Whether ``candidate`` may legally be appended to ``chain``.

    Rules:
    * Types are compatible (with ``"*"`` universal).
    * The candidate is not already used in the chain (no cycles).
    * We do not chain two "input" components in a row — this is the
      parsimony guard that prevents redundant no-op stages.
    """
    if not chain:
        return True
    last = chain[-1]
    if not matches(last.output, candidate.input):
        return False
    if any(c.id == candidate.id for c in chain):
        return False
    if last.type == "input" and candidate.type == "input":
        return False
    return True
