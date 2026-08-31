"""Modern workflow-graph renderer.

Draws layered DAGs (or linear chains) as flat, boxy nodes with subtle
type-coded left borders — the visual language of a modern diagram tool
rather than a colourful NetworkX default.

Design decisions:
    * Rounded rectangles, not circles — nodes read as labels, not blobs
    * Neutral fill (near-white) with a coloured left-edge stripe per type
    * Thin grey arrows that draw *between* nodes, not through them
    * System typography (Helvetica / Inter substitute) at readable size
    * No matplotlib legend — the type-coded stripes are self-explanatory
      and a legend just adds clutter
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from core.models import Architecture


# ------------------------------------------------------------------ palette
# Slightly desaturated pastels that read on a light background. The "stripe"
# on the left of each node is the strongly-coloured version.
_TYPE_STRIPE = {
    "input":      "#60A5FA",   # blue
    "retrieval":  "#FBBF24",   # amber
    "reasoning":  "#A78BFA",   # violet
    "validation": "#34D399",   # emerald
    "aggregator": "#22D3EE",   # cyan
    "output":     "#A1A1AA",   # slate
}

_NODE_FILL   = "#18181B"      # dark card
_NODE_BORDER = "#3F3F46"
_TEXT_COLOR  = "#FAFAFA"
_TEXT_MUTED  = "#A1A1AA"
_ARROW_COLOR = "#71717A"
_BG_COLOR    = "#0A0A0A"      # matches app bg


# ------------------------------------------------------------------ helpers
def _wrap(text: str, width: int = 18) -> str:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _layer_positions(arch: Architecture) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """Return (layer_index, position_in_layer) -> (x, y) coordinates."""
    positions: Dict[Tuple[int, int], Tuple[float, float]] = {}
    if arch.layers is not None:
        layers = arch.layers
    else:
        layers = [[c] for c in arch.components]

    n_layers = len(layers)
    for li, layer in enumerate(layers):
        y = -li * 1.6      # vertical spacing between layers
        width = len(layer)
        # Spread nodes horizontally within the layer.
        if width == 1:
            xs = [0.0]
        else:
            span = 2.6 * (width - 1) / 2
            xs = [-span + i * (span * 2 / (width - 1)) for i in range(width)]
        for xi, _ in enumerate(layer):
            positions[(li, xi)] = (xs[xi], y)
    return positions


# ------------------------------------------------------------------ drawing
def _draw_node(ax, x, y, w, h, label: str, stripe_color: str) -> None:
    # Main body — rounded rectangle
    body = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w, h,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor=_NODE_FILL,
        edgecolor=_NODE_BORDER,
        linewidth=1.1,
        zorder=2,
    )
    ax.add_patch(body)
    # Coloured left stripe — plain rectangle so it stays as a clean bar
    # (a FancyBboxPatch with rounding on such a thin shape renders as an oval)
    stripe_w = 0.08
    stripe = Rectangle(
        (x - w / 2 + 0.015, y - h / 2 + 0.08),
        stripe_w, h - 0.16,
        facecolor=stripe_color,
        edgecolor="none",
        zorder=3,
    )
    ax.add_patch(stripe)
    # Label — centered accounting for the stripe on the left
    ax.text(
        x + stripe_w / 2 + 0.03, y,
        _wrap(label, width=18),
        ha="center", va="center",
        fontsize=9, color=_TEXT_COLOR, fontweight="normal",
        family="sans-serif",
        zorder=4,
    )


def _draw_arrow(ax, x1, y1, x2, y2, node_w: float, node_h: float) -> None:
    """Draw an arrow between two node CENTRES, shortened at both ends so
    the head sits *outside* the destination node's border."""
    dx = x2 - x1
    dy = y2 - y1
    dist = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / dist, dy / dist

    # Project the intersection of the line with each rectangle. Since the
    # line is either near-vertical or diagonal, we approximate by using the
    # node's half-diagonal in the direction of travel.
    half_diag = (node_w ** 2 + node_h ** 2) ** 0.5 / 2
    # Shrink slightly more than half_diag so the arrowhead is fully outside.
    off_src = min(half_diag * 0.75, dist / 2 - 0.05)
    off_dst = min(half_diag * 0.75, dist / 2 - 0.05)

    sx = x1 + ux * off_src
    sy = y1 + uy * off_src
    ex = x2 - ux * off_dst
    ey = y2 - uy * off_dst

    arrow = FancyArrowPatch(
        (sx, sy), (ex, ey),
        arrowstyle="-|>",
        mutation_scale=18,
        color=_ARROW_COLOR, linewidth=1.3,
        shrinkA=0, shrinkB=0,
        zorder=1,
    )
    ax.add_patch(arrow)


# ------------------------------------------------------------------ main
def render_architecture(arch: Architecture, figsize=(7.5, None)):
    """Return a Matplotlib Figure showing the workflow top-to-bottom."""
    if arch.layers is not None:
        layers = arch.layers
    else:
        layers = [[c] for c in arch.components]

    n_layers = len(layers)
    max_width = max((len(l) for l in layers), default=1)

    # Dynamic figure height based on layer count.
    fig_h = figsize[1] if figsize[1] is not None else max(4.2, 1.05 * n_layers + 0.8)
    fig_w = figsize[0]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    positions = _layer_positions(arch)
    node_w = 2.05
    node_h = 0.72

    # Draw arrows first (behind nodes)
    for li in range(n_layers - 1):
        for xi_from, _ in enumerate(layers[li]):
            for xi_to, _ in enumerate(layers[li + 1]):
                x1, y1 = positions[(li, xi_from)]
                x2, y2 = positions[(li + 1, xi_to)]
                _draw_arrow(ax, x1, y1, x2, y2, node_w, node_h)

    # Draw nodes
    for li, layer in enumerate(layers):
        for xi, comp in enumerate(layer):
            x, y = positions[(li, xi)]
            stripe = _TYPE_STRIPE.get(comp.type, "#71717A")
            _draw_node(ax, x, y, node_w, node_h, comp.name, stripe)

    # Compute limits with padding
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    pad_x = 2.0
    pad_y = 1.2
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)
    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.tight_layout(pad=0.4)
    return fig
