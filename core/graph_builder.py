"""Helpers for turning an :class:`Architecture` chain into a NetworkX graph."""

from __future__ import annotations

import networkx as nx

from .models import Architecture


def build_graph(arch: Architecture) -> nx.DiGraph:
    """Return a directed graph describing the workflow.

    Each node carries the component's metadata as attributes so that the
    visualisation layer can colour and label nodes without extra lookups.
    """
    g = nx.DiGraph()
    for idx, comp in enumerate(arch.components):
        node_id = f"{idx}_{comp.id}"
        g.add_node(
            node_id,
            label=comp.name,
            type=comp.type,
            cost=comp.cost,
            latency=comp.latency,
            memory=comp.memory,
            quality=comp.quality,
        )
        if idx > 0:
            prev = f"{idx-1}_{arch.components[idx-1].id}"
            g.add_edge(prev, node_id)
    return g
