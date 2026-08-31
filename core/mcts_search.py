"""Monte-Carlo Tree Search architecture engine.

An alternative to :mod:`core.beam_search`. Where beam search greedily keeps
the top-K partial pipelines at every depth, MCTS balances **exploration**
(sampling under-visited components) against **exploitation** (deepening
promising branches) through the UCT (Upper Confidence bound applied to
Trees) formula:

    UCT(child) = mean_reward(child) + c * sqrt( ln(parent.visits) / child.visits )

MCTS is useful when:

* the component repository is large and beam search's top-K prunes too
  aggressively;
* the scoring landscape is non-monotonic (a short chain looks poor at depth
  2 but becomes optimal after a validator is added);
* the search budget is measured in iterations rather than depth, which maps
  cleanly onto an interactive user's "search harder" button.

The public entrypoint :func:`mcts_search` has the same signature as
:func:`core.beam_search.beam_search` so the two engines are interchangeable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .models import Architecture, Component, Constraints, Task
from .repository import ComponentRepository
from .scorer import compute_metrics, rank, satisfies_constraints


# --------------------------------------------------------------------------- config
_MAX_DEPTH = 15
_TERMINAL_OUTPUTS = {"json", "report"}
_UCT_C = math.sqrt(2)          # classical exploration constant
_DEFAULT_ITERATIONS = 400       # cheap for small repos; tune for larger ones
_INVALID_REWARD = 0.0           # reward for a rollout that dead-ends
_DETERMINISTIC_SEED = 1729      # so a demo yields the same tree each run


# --------------------------------------------------------------------------- helpers
def _valid_starts(repo: ComponentRepository, task: Task) -> List[Component]:
    inputs = repo.by_type(task.domain, "input")
    return [c for c in inputs if c.input == task.input_type] or inputs


def _successors(
    repo: ComponentRepository,
    arch: Architecture,
    task: Task,
    cons: Constraints,
) -> List[Component]:
    """Compatible next components respecting types, cycles and CPU flag."""
    last = arch.components[-1]
    used: Set[str] = {c.id for c in arch.components}
    return [
        c for c in repo.get(task.domain)
        if c.id not in used
        and c.input == last.output
        and (not cons.cpu_only or c.cpu)
        # Parsimony: never chain two "input" components in a row.
        and not (last.type == "input" and c.type == "input")
    ]


def _is_valid_terminal(arch: Architecture, cons: Constraints) -> bool:
    if not arch.components:
        return False
    tail = arch.components[-1]
    if tail.type != "output" or tail.output not in _TERMINAL_OUTPUTS:
        return False
    return satisfies_constraints(arch, cons)


def _over_budget(arch: Architecture, cons: Constraints) -> bool:
    return (
        arch.total_cost > cons.max_cost
        or arch.total_latency > cons.max_latency
        or arch.peak_memory > cons.max_memory
    )


# --------------------------------------------------------------------------- node
@dataclass
class _Node:
    """A node in the search tree.

    Each node owns the partial :class:`Architecture` that leads to it, so
    reconstructing a full pipeline never requires walking back to the root.
    """
    arch: Architecture
    parent: Optional["_Node"] = None
    children: Dict[str, "_Node"] = field(default_factory=dict)
    untried: List[Component] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0

    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def is_fully_expanded(self) -> bool:
        return not self.untried

    def best_child(self, c: float = _UCT_C) -> "_Node":
        """Pick the child that maximises the UCT criterion."""
        log_n = math.log(max(self.visits, 1))
        best, best_score = None, -math.inf
        for child in self.children.values():
            if child.visits == 0:
                # Guarantee every child gets tried once before UCT kicks in.
                return child
            exploit = child.mean_reward()
            explore = c * math.sqrt(log_n / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score, best = score, child
        return best  # type: ignore[return-value]


# --------------------------------------------------------------------------- search
def mcts_search(
    repo: ComponentRepository,
    task: Task,
    cons: Constraints,
    iterations: int = _DEFAULT_ITERATIONS,
    top_k: int = 3,
    exploration: float = _UCT_C,
    seed: Optional[int] = _DETERMINISTIC_SEED,
) -> List[Architecture]:
    """Run MCTS and return up to ``top_k`` best terminal architectures.

    Parameters
    ----------
    repo, task, cons
        Same meaning as in :func:`core.beam_search.beam_search`.
    iterations : int
        Number of full select-expand-simulate-backpropagate cycles.
    top_k : int
        How many completed pipelines to return.
    exploration : float
        The UCT ``c`` constant. Larger values push the search wider; the
        default ``sqrt(2)`` is the classical value from Kocsis & Szepesvari.
    seed : int, optional
        Fixed RNG seed for reproducible demos; pass ``None`` for stochastic
        runs.
    """
    rng = random.Random(seed)

    # ---- root: a virtual "empty pipeline" whose children are valid inputs
    root = _Node(arch=Architecture(components=[]))
    root.untried = list(_valid_starts(repo, task))

    completed: Dict[str, Architecture] = {}

    for _ in range(iterations):
        node = _select(root, exploration)

        # ---- expansion
        if node.untried and len(node.arch.components) < _MAX_DEPTH:
            comp = node.untried.pop(rng.randrange(len(node.untried)))
            child_arch = Architecture(components=node.arch.components + [comp])
            compute_metrics(child_arch)

            if _over_budget(child_arch, cons):
                # Skip: creating this child is a guaranteed dead end.
                _backpropagate(node, _INVALID_REWARD)
                continue

            child = _Node(arch=child_arch, parent=node)
            child.untried = _successors(repo, child_arch, task, cons)
            node.children[comp.id] = child
            node = child

        # ---- simulation (random rollout to a terminal or dead-end)
        reward, terminal = _rollout(node.arch, repo, task, cons, rng)
        if terminal is not None:
            completed.setdefault(terminal.signature(), terminal)

        # ---- backpropagation
        _backpropagate(node, reward)

    # Also record any terminal already sitting in the tree.
    _collect_tree_terminals(root, cons, completed)

    return rank(list(completed.values()))[:top_k]


# --------------------------------------------------------------------------- phases
def _select(root: _Node, c: float) -> _Node:
    """Descend from the root along the UCT-best child until we hit a node
    that still has untried moves or has no children (a terminal / dead-end)."""
    node = root
    while node.is_fully_expanded() and node.children:
        node = node.best_child(c)
    return node


def _rollout(
    arch: Architecture,
    repo: ComponentRepository,
    task: Task,
    cons: Constraints,
    rng: random.Random,
) -> tuple[float, Optional[Architecture]]:
    """Simulate a random continuation until we terminate or dead-end.

    Returns
    -------
    (reward, terminal)
        ``reward`` is the score of the resulting architecture (or
        ``_INVALID_REWARD`` if the rollout dead-ended). ``terminal`` is the
        completed :class:`Architecture` if the rollout succeeded, else None.
    """
    current = Architecture(components=list(arch.components))
    compute_metrics(current)

    # Already a valid terminal? Score it as-is.
    if _is_valid_terminal(current, cons):
        return current.score, current

    for _ in range(_MAX_DEPTH - len(current.components)):
        if not current.components:
            # Should not happen: root rollouts occur only via expansion.
            return _INVALID_REWARD, None
        succs = _successors(repo, current, task, cons)
        if not succs:
            return _INVALID_REWARD, None

        # Slightly bias the random policy toward higher-quality components so
        # rollouts do not degenerate into random noise. Weight = quality**2.
        weights = [max(c.quality, 0.01) ** 2 for c in succs]
        nxt = rng.choices(succs, weights=weights, k=1)[0]

        current = Architecture(components=current.components + [nxt])
        compute_metrics(current)

        if _over_budget(current, cons):
            return _INVALID_REWARD, None
        if _is_valid_terminal(current, cons):
            return current.score, current

    return _INVALID_REWARD, None


def _backpropagate(node: Optional[_Node], reward: float) -> None:
    while node is not None:
        node.visits += 1
        node.total_reward += reward
        node = node.parent


def _collect_tree_terminals(
    node: _Node,
    cons: Constraints,
    out: Dict[str, Architecture],
) -> None:
    """DFS the tree collecting any node that is already a valid terminal."""
    if _is_valid_terminal(node.arch, cons):
        out.setdefault(node.arch.signature(), node.arch)
    for c in node.children.values():
        _collect_tree_terminals(c, cons, out)
