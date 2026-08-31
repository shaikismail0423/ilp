"""Synthetic BPI-Challenge-style task generator.

The BPI Challenge datasets (2012, 2017, 2020) contain real business-process
event logs. Each process instance is a DAG of activities with dependencies
and completion outcomes. This module produces *synthetic* task DAGs with the
same structure so AutoAgentSearch can be benchmarked without an internet
download.

A generated task is described by:

* ``task_id``    - a stable identifier
* ``label``      - human-readable name (e.g. "Approve Loan")
* ``nodes``      - list of activity names
* ``edges``      - list of ``(src, dst)`` dependency pairs
* ``difficulty`` - float in [0, 1]; harder tasks need more validation
* ``domain``     - always ``"agentic"`` for BPI-derived tasks
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class BPITask:
    task_id: str
    label: str
    nodes: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    difficulty: float = 0.5
    domain: str = "agentic"

    def as_task_text(self) -> str:
        """A one-line description usable by ``core.task_parser.parse_task``."""
        return (
            f"Execute the {self.label} business process across "
            f"{len(self.nodes)} activities and produce a status report."
        )


# ---------------------------------------------------------------- templates
_TEMPLATES = [
    {
        "label": "Approve Loan",
        "nodes": ["Submit Form", "Credit Check", "Review", "Approval", "Notification"],
        "edges": [("Submit Form", "Credit Check"),
                  ("Credit Check", "Review"),
                  ("Review", "Approval"),
                  ("Approval", "Notification")],
    },
    {
        "label": "Onboard Customer",
        "nodes": ["KYC Submission", "Identity Check", "Risk Scoring",
                  "Account Creation", "Welcome"],
        "edges": [("KYC Submission", "Identity Check"),
                  ("Identity Check", "Risk Scoring"),
                  ("Risk Scoring", "Account Creation"),
                  ("Account Creation", "Welcome")],
    },
    {
        "label": "Process Insurance Claim",
        "nodes": ["Intake", "Document Verification", "Adjuster Review",
                  "Payment Decision", "Payout"],
        "edges": [("Intake", "Document Verification"),
                  ("Document Verification", "Adjuster Review"),
                  ("Adjuster Review", "Payment Decision"),
                  ("Payment Decision", "Payout")],
    },
    {
        "label": "Resolve Support Ticket",
        "nodes": ["Ticket Opened", "Triage", "Investigation",
                  "Resolution", "Closure"],
        "edges": [("Ticket Opened", "Triage"),
                  ("Triage", "Investigation"),
                  ("Investigation", "Resolution"),
                  ("Resolution", "Closure")],
    },
    {
        "label": "Fulfil Purchase Order",
        "nodes": ["Order Placed", "Stock Check", "Payment Captured",
                  "Shipment Prepared", "Delivered"],
        "edges": [("Order Placed", "Stock Check"),
                  ("Stock Check", "Payment Captured"),
                  ("Payment Captured", "Shipment Prepared"),
                  ("Shipment Prepared", "Delivered")],
    },
]


def generate_tasks(n: int = 20, seed: int = 42) -> List[BPITask]:
    """Generate ``n`` BPI-style tasks with reproducible variation."""
    rng = random.Random(seed)
    out: List[BPITask] = []
    for i in range(n):
        tmpl = _TEMPLATES[i % len(_TEMPLATES)]
        # add mild variation: drop 0-1 optional node, jitter difficulty
        nodes = list(tmpl["nodes"])
        edges = list(tmpl["edges"])
        difficulty = rng.uniform(0.25, 0.85)
        out.append(BPITask(
            task_id=f"bpi_{i:03d}",
            label=tmpl["label"] + (f" (variant {i // len(_TEMPLATES) + 1})"
                                   if i >= len(_TEMPLATES) else ""),
            nodes=nodes,
            edges=edges,
            difficulty=difficulty,
        ))
    return out
