# AutoAgentSearch — Extended Problem Statement

*A research-quality framing suitable for a TCS Research prototype and for
enterprise pilot conversations.*

---

## 1. Original problem (recap)

Enterprises in regulated verticals — healthcare, finance, legal — want to
automate document- and text-heavy tasks with AI. The current practice is
that a solutions engineer manually assembles a pipeline (OCR, parser,
model, validator, formatter) for every new task and every new customer.
This is slow, expensive, non-repeatable, and heavily dependent on the
engineer's biases about which models to trust.

**AutoAgentSearch v1** replaces that manual step with a *search*: given a
task description and a resource budget, the system explores a repository
of reusable AI components and returns the best pipeline it can find.

## 2. Why the original framing is not enough

The v1 formulation optimises a single scalar (quality − cost − latency −
memory) over a linear chain of components under a single user. Real
enterprise deployments diverge from that in five important ways:

1. **Multiple, competing stakeholders.** The compliance officer, the CFO,
   the ML platform lead and the end user each have different priorities.
   A single scalar score hides those trade-offs.
2. **Uncertainty in component metadata.** Latency, cost and quality are
   never point estimates in production — they drift with input
   distribution, model version and load. A workflow that wins on paper
   can be Pareto-dominated once uncertainty is modelled.
3. **Compliance is a hard constraint, not a soft cost.** HIPAA / PCI / data
   residency / on-prem-only rules cannot be traded off against a small
   quality gain the way cost can.
4. **Real tasks are DAGs, not chains.** Extracting *diagnosis* **and**
   *medication* **and** *ICD codes* is three parallel sub-pipelines whose
   outputs fuse into one JSON — a linear beam-search chain cannot express
   that.
5. **Metadata gets stale.** Any prediction the search made must be checked
   against what the deployed pipeline actually does, and the difference
   must feed back into the repository or the search silently drifts from
   reality.

## 3. Extended problem statement

Let a **task** *T* consist of:

* a natural-language description,
* a **domain** *d* ∈ {healthcare, finance, …},
* a set of **sub-goals** *G* = {g₁, …, gₖ} (each an intent + desired output),
* a set of **hard constraints** *H* (regulatory, deployment, data-locality)
  that any pipeline **must** satisfy, and
* a set of **soft objectives** *O* = {o₁, …, oₘ} with per-stakeholder weights
  (cost, latency, memory, quality, reliability, explainability, carbon).

Let a **component repository** *R* be a set of typed components, each
described by:

* a nominal metadata tuple *(cost, latency, memory, quality, …)*, and
* a **confidence distribution** over each metric (e.g. a mean and a
  standard deviation, or a small empirical sample from prior runs).

The **extended search problem** is: find a directed acyclic graph *A*
over components in *R* such that

* every sub-goal in *G* is produced by some sink of *A*,
* every hard constraint in *H* is satisfied by every path in *A*,
* the expected soft-objective vector *E[o(A)]* lies on the Pareto frontier
  of feasible workflows under the metadata uncertainty, and
* the *worst-case* (e.g. 95th-percentile) soft-objective vector still lies
  within the user's declared risk tolerance.

The system returns not a single winner but the Pareto frontier, with each
point annotated by which stakeholder profile it best serves and by which
metadata assumptions its ranking depends on.

## 4. Concrete extensions on top of the v1 code base

The v1 code (beam search + scalar score + linear chain + point metadata) is
the correct starting scaffold. The extensions below are additive; the
search engine's public signature does not need to change.

| # | Extension | What it adds | Where it plugs in |
|---|-----------|-------------|-------------------|
| 1 | **MCTS engine** (this release) | Balanced exploration when the repo grows and beam pruning is too aggressive. | `core/mcts_search.py`, selectable in the UI. |
| 2 | **Metadata with uncertainty** | Every metric becomes a `(mean, std)` pair; the scorer aggregates expectations and reports confidence bands. | `core/models.py` (Component), `core/scorer.py`. |
| 3 | **Pareto-front output** | Return the non-dominated set rather than a scalar top-3. | New `core/pareto.py`; the UI plots the frontier. |
| 4 | **Multi-stakeholder profiles** | Named weight vectors (Compliance-first, Cost-first, Quality-first) picked in the sidebar. | `core/models.py` (Profile), `core/scorer.py`. |
| 5 | **Hard-constraint DSL** | Rules like "no component with `pii=true` may run off-prem" or "must be HIPAA-certified" evaluated *before* scoring. | New `core/constraints_dsl.py`. |
| 6 | **DAG sub-goal decomposition** | Task Parser returns a set of sub-goals; the search composes parallel pipelines that fuse into a single output stage. | Extend `core/graph_builder.py` and add a DAG-aware terminal check. |
| 7 | **Closed-loop metadata** | After the winning pipeline runs, measured cost/latency/quality feed back into the repository (Bayesian update on the metric distributions). | New `core/telemetry.py`; nightly job re-writes JSON. |
| 8 | **Explainability & audit trail** | Every returned pipeline carries the list of pruned alternatives, the metric bands, and the constraint checks it passed. | Extend `core/explainer.py`. |
| 9 | **Human-in-the-loop stages** | A component may declare `requires_review=true`; the pipeline emits an approval task before continuing. | Metadata schema + a new `human_review` component type. |
| 10 | **Cost/quality Pareto learning** | Replace the analytical scorer with a learned surrogate trained on past pipeline runs (a small XGBoost or GP) once telemetry exists. | New `core/surrogate.py`; scorer becomes an adapter. |

## 5. Success criteria

A future version of AutoAgentSearch is successful when it:

1. Recovers the same pipeline a senior ML engineer would design, **and**
   surfaces at least one plausible alternative the engineer had not
   considered (measured on a benchmark of 20 real enterprise tasks per
   domain).
2. Its predicted metrics are within a documented tolerance of the actual
   post-deployment measurements (e.g. ± 20 % latency, ± 15 % cost).
3. Every returned pipeline is auditable: the compliance officer can read
   *why* it satisfies HIPAA without running the code.
4. Adding a new domain requires only a new component-repository JSON —
   zero code changes to the engine.

## 6. Non-goals

* AutoAgentSearch does not train new models.
* It does not host or execute the returned pipeline — that is a job for
  the deployment platform. The search is the *design-time* system.
* It does not attempt to select the best prompt for a given model. Prompt
  optimisation is a component-internal concern.

## 7. Research questions this framing opens

* **Q1**  How do you calibrate a component's quality distribution from a
  small handful of runs so that the search does not over- or under-trust
  new components?
* **Q2**  Which search strategy — beam, MCTS, evolutionary, learned policy
  — dominates as the repository size grows past ~100 components?
* **Q3**  Can the surrogate scorer transfer across domains, or must every
  vertical maintain its own?
* **Q4**  How do you make the Pareto frontier legible to a business user
  who thinks in "cheap / fast / accurate" rather than in vectors?
