"""Reinforcement-learning-based architecture search.

A tabular Q-learning agent explores the space of valid workflows and
learns which component to pick given the current pipeline's tail output.

State
-----
The state is the output type of the last-appended component, or the
special sentinel ``"_start_"`` when the pipeline is empty. This is a
coarse but useful abstraction — it collapses many concrete partial
pipelines into a small number of states, which is what makes tabular
Q-learning tractable on our repositories.

Action
------
An action is *"append this component next"*. The eligible action set at
each state is exactly the set of components whose ``input`` matches the
state (plus the usual filters — not already used, honours CPU-only, keeps
the pipeline in-budget after the addition).

Reward
------
* +score of the finished architecture on reaching a valid terminal
* 0 for every non-terminal transition (Q-learning bootstraps the value)
* -1 for a transition that pushes the pipeline over its resource budget
* -0.5 for a rollout that runs out of depth without reaching a terminal

Standard Q-learning update:

    Q(s, a) <- Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]

After ``episodes`` training rollouts we do a greedy rollout for the final
result. Every valid terminal architecture seen during training is also
kept and returned in the top-K, so RL surfaces multiple candidates even
when the greedy policy converges to one.

Like the other engines, ``rl_search`` has the same signature so beam,
MCTS and RL are interchangeable.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .models import Architecture, Component, Constraints, Task
from .repository import ComponentRepository
from .scorer import compute_metrics, rank, satisfies_constraints


# --------------------------------------------------------------------------- config
_MAX_DEPTH = 15
_TERMINAL_OUTPUTS = {"json", "report"}
_START_STATE = "_start_"

_DEFAULT_EPISODES   = 300
_DEFAULT_ALPHA      = 0.25   # learning rate
_DEFAULT_GAMMA      = 0.90   # discount factor
_DEFAULT_EPSILON0   = 0.40   # initial exploration
_DEFAULT_EPSILON_MIN = 0.05  # final exploration (linearly annealed)
_DEFAULT_SEED       = 2023


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


def _eligible(
    repo: ComponentRepository,
    arch: Architecture,
    task: Task,
    cons: Constraints,
) -> List[Component]:
    """The action set from the current pipeline's state."""
    if not arch.components:
        return _valid_starts(repo, task)
    return _successors(repo, arch, task, cons)


def _is_terminal(arch: Architecture, cons: Constraints) -> bool:
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


def _state_of(arch: Architecture) -> str:
    return arch.components[-1].output if arch.components else _START_STATE


# --------------------------------------------------------------------------- policy
def _choose_action(
    Q: Dict[Tuple[str, str], float],
    actions: List[Component],
    state: str,
    epsilon: float,
    rng: random.Random,
) -> Component:
    """ε-greedy over Q(s, a). Ties broken by RNG for reproducible fairness."""
    if rng.random() < epsilon:
        return rng.choice(actions)
    # Argmax with random tie-break.
    best_val = max(Q[(state, a.id)] for a in actions)
    best = [a for a in actions if Q[(state, a.id)] == best_val]
    return rng.choice(best)


# --------------------------------------------------------------------------- search
def rl_search(
    repo: ComponentRepository,
    task: Task,
    cons: Constraints,
    episodes: int = _DEFAULT_EPISODES,
    top_k: int = 3,
    alpha: float = _DEFAULT_ALPHA,
    gamma: float = _DEFAULT_GAMMA,
    epsilon: float = _DEFAULT_EPSILON0,
    seed: Optional[int] = _DEFAULT_SEED,
) -> List[Architecture]:
    """Train a Q-learning policy and return the top-K terminals it found."""
    rng = random.Random(seed)
    Q: Dict[Tuple[str, str], float] = defaultdict(float)
    completed: Dict[str, Architecture] = {}

    for ep in range(episodes):
        # Anneal exploration linearly from ε₀ to ε_min over training.
        frac = ep / max(episodes - 1, 1)
        eps = epsilon * (1 - frac) + _DEFAULT_EPSILON_MIN * frac

        arch = Architecture(components=[])
        state = _state_of(arch)
        # Track the most recent (state-before-action, action-id) pair so the
        # depth-exhaustion penalty at the end of the episode credits the
        # correct Q-cell.
        last_pre_state: Optional[str] = None
        last_action_id: Optional[str] = None

        for _step in range(_MAX_DEPTH):
            actions = _eligible(repo, arch, task, cons)
            if not actions:
                # Dead-end from this state — small penalty to whatever choice
                # brought us here has already been applied.
                break

            action = _choose_action(Q, actions, state, eps, rng)
            new_arch = Architecture(components=arch.components + [action])
            compute_metrics(new_arch)

            if _over_budget(new_arch, cons):
                # Penalise this action strongly so the policy learns to avoid
                # it from the same state next time.
                Q[(state, action.id)] += alpha * (-1.0 - Q[(state, action.id)])
                break

            next_state = action.output
            if _is_terminal(new_arch, cons):
                reward = new_arch.score
                # Terminal — no future value to bootstrap from.
                Q[(state, action.id)] += alpha * (reward - Q[(state, action.id)])
                completed.setdefault(new_arch.signature(), new_arch)
                break

            # Non-terminal step: bootstrap using max Q over the next state's
            # eligible actions.
            future_actions = _eligible(repo, new_arch, task, cons)
            future = (
                max(Q[(next_state, fa.id)] for fa in future_actions)
                if future_actions else 0.0
            )
            Q[(state, action.id)] += alpha * (
                0.0 + gamma * future - Q[(state, action.id)]
            )

            last_pre_state = state
            last_action_id = action.id
            arch = new_arch
            state = next_state
        else:
            # Ran out of depth without terminating — mild penalty on the
            # last chosen (state, action) pair so the policy learns not to
            # ramble. Uses the pre-action state, not the post-action state.
            if last_pre_state is not None and last_action_id is not None:
                Q[(last_pre_state, last_action_id)] += alpha * (
                    -0.5 - Q[(last_pre_state, last_action_id)]
                )

    # ---- greedy rollout using the learned policy
    arch = Architecture(components=[])
    state = _state_of(arch)
    for _ in range(_MAX_DEPTH):
        actions = _eligible(repo, arch, task, cons)
        if not actions:
            break
        best = max(actions, key=lambda a: Q[(state, a.id)])
        new_arch = Architecture(components=arch.components + [best])
        compute_metrics(new_arch)
        if _over_budget(new_arch, cons):
            break
        arch = new_arch
        state = best.output
        if _is_terminal(arch, cons):
            completed.setdefault(arch.signature(), arch)
            break

    return rank(list(completed.values()))[:top_k]
