"""Rule-based task parser.

Turns a free-form business request into a structured :class:`Task`.
No LLM inference is performed; matching is purely keyword-driven.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .models import Task


# ---------------------------------------------------------------------------
# Keyword tables. Ordered by specificity - the most specific intents come
# first so that "loan risk" is matched before the generic "loan".
# ---------------------------------------------------------------------------
_HEALTHCARE_INTENTS: List[Tuple[str, List[str]]] = [
    ("icd_coding",           ["icd", "coding", "diagnosis code"]),
    ("medication_extraction", ["medication", "drug", "prescription", "med list"]),
    ("diagnosis_extraction", ["diagnosis", "diagnos", "disease", "condition"]),
    ("patient_summary",      ["summary", "summarize", "patient", "discharge"]),
]

_FINANCE_INTENTS: List[Tuple[str, List[str]]] = [
    ("fraud_detection",      ["fraud", "suspicious", "anomaly", "aml"]),
    ("loan_risk",            ["loan", "credit", "risk", "underwriting"]),
    ("invoice_analysis",     ["invoice", "receipt", "purchase order", "bill"]),
    ("financial_report",     ["report", "statement", "earnings", "financial"]),
]

_DOMAIN_HINTS: Dict[str, List[str]] = {
    "healthcare": [
        "patient", "medical", "clinical", "diagnos", "medication",
        "hospital", "discharge", "icd", "doctor", "prescription", "ehr",
    ],
    "finance":   [
        "loan", "invoice", "fraud", "credit", "bank", "finance",
        "transaction", "payment", "portfolio", "audit",
    ],
    # The "agentic" library is used internally by the orchestration wrapper,
    # not selected by user-facing task text. The benchmark constructs its
    # own Task objects and does not go through auto-detection.
}

_INPUT_HINTS: Dict[str, List[str]] = {
    "document": ["pdf", "scan", "image", "document", "form", "invoice", "report"],
    "text":     ["note", "text", "email", "message", "summary", "log"],
}

_OUTPUT_HINTS: Dict[str, List[str]] = {
    "report": ["report", "summary", "dashboard", "assessment"],
    "json":   ["json", "structured", "fields", "extract"],
}


def _match_first(text: str, table: Dict[str, List[str]], default: str) -> str:
    text = text.lower()
    scores: Dict[str, int] = {k: 0 for k in table}
    for key, keywords in table.items():
        for kw in keywords:
            if kw in text:
                scores[key] += 1
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else default


def _match_intent(text: str, intents: List[Tuple[str, List[str]]], default: str) -> str:
    text = text.lower()
    for intent, keywords in intents:
        if any(kw in text for kw in keywords):
            return intent
    return default


def parse_task(text: str, explicit_domain: str = "auto") -> Task:
    """Parse a business request into a structured Task.

    Parameters
    ----------
    text : str
        The raw task description entered by the user.
    explicit_domain : str
        If not "auto", overrides the domain detected from the text.
    """
    text = (text or "").strip()

    if explicit_domain and explicit_domain != "auto":
        domain = explicit_domain
    else:
        domain = _match_first(text, _DOMAIN_HINTS, default="healthcare")

    if domain == "healthcare":
        intent = _match_intent(text, _HEALTHCARE_INTENTS, "diagnosis_extraction")
        default_output = "json"
    elif domain == "finance":
        intent = _match_intent(text, _FINANCE_INTENTS, "fraud_detection")
        default_output = "report"
    else:  # agentic
        intent = "process_execution"
        default_output = "report"

    if domain == "agentic":
        input_type = "task"
    else:
        input_type = _match_first(text, _INPUT_HINTS, default="document")
    desired_output = _match_first(text, _OUTPUT_HINTS, default=default_output)

    return Task(
        domain=domain,
        intent=intent,
        input_type=input_type,
        desired_output=desired_output,
        raw=text,
    )
