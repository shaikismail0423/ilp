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


_UNIVERSAL = "*"


def matches(produced: str, expected: str) -> bool:
    """Return True if a stage producing ``produced`` can feed one expecting
    ``expected``. ``"*"`` on either side is universal."""
    return produced == expected or produced == _UNIVERSAL or expected == _UNIVERSAL
