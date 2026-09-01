"""Lightweight agent simulator.

The PDF blueprint recommends: "Instead of real agents, each agent performs a
function." This module implements exactly that. Every component id maps to a
Python function that:

* consumes the current pipeline state,
* mutates it (marks a plan as scheduled, adds a verification stamp, etc.),
* returns a ``bool`` indicating stage-level success or failure,
* deterministically simulates its cost/latency from its metadata plus the
  task's difficulty and any injected failures.

This is the middle ground between "metadata-only scoring" (fast but
opinion-based) and "run a real LLM" (accurate but prohibitively expensive
inside a search loop). It gives us a *measured* success rate to report in
the benchmark rather than relying only on the analytical score.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .bpi_tasks import BPITask
from .models import Architecture, Component


# --------------------------------------------------------------------------- state
@dataclass
class SimState:
    """Mutable state passed down the pipeline during execution."""
    task: BPITask
    stage_outputs: List[str] = field(default_factory=list)
    plan_built: bool = False
    plan_scheduled: bool = False
    plan_verified: bool = False
    executed: bool = False
    recovered: bool = False
    failures: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- helpers
def _prob_success(comp: Component, task: BPITask, boost: float = 0.0) -> float:
    """A stage's success probability = its declared quality, modulated by
    the task's difficulty and any boost from an earlier verifier."""
    return max(0.0, min(1.0, comp.quality * (1.1 - task.difficulty) + boost))


# --------------------------------------------------------------------------- agent fns
def _run_planner(state: SimState, comp: Component, rng: random.Random) -> bool:
    p = _prob_success(comp, state.task)
    ok = rng.random() < p
    state.plan_built = ok
    if not ok:
        state.failures.append("planner_failed")
    return ok


def _run_coordinator(state: SimState, comp: Component, rng: random.Random) -> bool:
    if not state.plan_built:
        return False
    p = _prob_success(comp, state.task)
    ok = rng.random() < p
    state.plan_scheduled = ok
    return ok


def _run_resource_manager(state: SimState, comp: Component, rng: random.Random) -> bool:
    if not state.plan_built:
        return False
    p = _prob_success(comp, state.task)
    ok = rng.random() < p
    state.plan_scheduled = ok
    return ok


def _run_verifier(state: SimState, comp: Component, rng: random.Random) -> bool:
    p = _prob_success(comp, state.task)
    ok = rng.random() < p
    state.plan_verified = ok
    return ok


def _run_executor(state: SimState, comp: Component, rng: random.Random) -> bool:
    # A verified plan is measurably more likely to execute correctly.
    boost = 0.08 if state.plan_verified else 0.0
    p = _prob_success(comp, state.task, boost=boost)
    ok = rng.random() < p
    state.executed = ok
    if not ok:
        state.failures.append("executor_failed")
    return ok


def _run_recovery(state: SimState, comp: Component, rng: random.Random) -> bool:
    """Recovery only helps if execution actually happened but produced errors,
    or was previously a failure that recovery can retry."""
    if state.executed:
        # Confirm the good result.
        state.recovered = True
        return True
    # Attempt to salvage a failed execution.
    p = _prob_success(comp, state.task) * 0.75
    ok = rng.random() < p
    state.executed = ok
    state.recovered = ok
    return ok


def _run_report(state: SimState, comp: Component, rng: random.Random) -> bool:
    # Reporting is nearly always successful; failure only if nothing to report.
    if not state.executed:
        return False
    return rng.random() < comp.quality


# --------------------------------------------------------------------------- registry
_AGENT_FNS: Dict[str, Callable[[SimState, Component, random.Random], bool]] = {
    "planner":               _run_planner,
    "coordinator":           _run_coordinator,
    "resource_manager":      _run_resource_manager,
    "verifier":              _run_verifier,
    "verifier_of_scheduled": _run_verifier,
    "executor":              _run_executor,
    "executor_raw":          _run_executor,
    "executor_deterministic":_run_executor,
    "executor_speculative":  _run_executor,
    "result_aggregator":     _run_executor,   # confirms executed state
    "recovery_agent":        _run_recovery,
    "status_report":         _run_report,
    "status_report_raw":     _run_report,
}


# --------------------------------------------------------------------------- run
@dataclass
class RunResult:
    """The output of one simulated pipeline run on one task."""
    success: bool
    stage_successes: List[bool]
    latency: float
    cost: float
    peak_memory: int
    n_agents: int
    recovered: bool
    failures: List[str]


def simulate_pipeline(
    arch: Architecture,
    task: BPITask,
    seed: Optional[int] = None,
    inject_failure: Optional[str] = None,
) -> RunResult:
    """Execute ``arch`` on ``task`` and return a :class:`RunResult`.

    Parameters
    ----------
    arch
        The architecture to run.
    task
        The BPI-style task the pipeline should complete.
    seed
        RNG seed for reproducible runs. ``None`` = fully stochastic.
    inject_failure
        Optional name of a failure to inject before execution:

        - ``"missing_activity"``: drops one activity from the task DAG,
          reducing every stage's success probability.
        - ``"incorrect_dependency"``: raises task difficulty temporarily.
        - ``"resource_shortage"``: halves the executor's success probability.
    """
    rng = random.Random(seed)

    if inject_failure == "missing_activity" and len(task.nodes) > 2:
        task = BPITask(
            task_id=task.task_id, label=task.label,
            nodes=task.nodes[:-1], edges=task.edges,
            difficulty=min(1.0, task.difficulty + 0.15),
        )
    elif inject_failure == "incorrect_dependency":
        task = BPITask(
            task_id=task.task_id, label=task.label,
            nodes=task.nodes, edges=task.edges,
            difficulty=min(1.0, task.difficulty + 0.20),
        )

    state = SimState(task=task)
    stage_ok: List[bool] = []
    latency = 0.0
    cost = 0.0
    peak_memory = 0

    for comp in arch.components:
        fn = _AGENT_FNS.get(comp.id)
        if fn is None:
            # Non-agentic components (healthcare / finance) succeed by
            # metadata quality directly — a graceful degradation.
            ok = rng.random() < comp.quality
        else:
            if inject_failure == "resource_shortage" and comp.id.startswith("executor"):
                # Halve success probability by scaling declared quality.
                weakened = Component(
                    id=comp.id, name=comp.name, domain=comp.domain, type=comp.type,
                    input=comp.input, output=comp.output,
                    latency=comp.latency, cost=comp.cost, memory=comp.memory,
                    quality=comp.quality * 0.5, cpu=comp.cpu,
                )
                ok = fn(state, weakened, rng)
            else:
                ok = fn(state, comp, rng)

        stage_ok.append(ok)
        latency += comp.latency
        cost += comp.cost
        peak_memory = max(peak_memory, comp.memory)

    return RunResult(
        success=state.executed and (arch.components[-1].type == "output"),
        stage_successes=stage_ok,
        latency=latency,
        cost=cost,
        peak_memory=peak_memory,
        n_agents=len(arch.components),
        recovered=state.recovered,
        failures=state.failures,
    )
