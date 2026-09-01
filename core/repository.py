"""Component repository.

Loads the JSON metadata files under ``components/`` and exposes the pool of
:class:`Component` objects available to the search engine.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from .models import Component


class ComponentRepository:
    """Loads and serves components grouped by domain."""

    def __init__(self, components_dir: str):
        self.components_dir = components_dir
        self._by_domain: Dict[str, List[Component]] = {}
        self._load_all()

    # ------------------------------------------------------------------ load
    def _load_all(self) -> None:
        if not os.path.isdir(self.components_dir):
            raise FileNotFoundError(
                f"Components directory not found: {self.components_dir}"
            )
        for fname in os.listdir(self.components_dir):
            if not fname.endswith(".json"):
                continue
            domain = fname.replace(".json", "")
            with open(os.path.join(self.components_dir, fname), "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._by_domain[domain] = [Component.from_dict(x) for x in raw]

    # -------------------------------------------------------------- queries
    def domains(self) -> List[str]:
        return sorted(self._by_domain.keys())

    def get(self, domain: str) -> List[Component]:
        return list(self._by_domain.get(domain, []))

    def by_type(self, domain: str, type_: str) -> List[Component]:
        return [c for c in self.get(domain) if c.type == type_]
