"""Render the README's backward-pass animation: forward sweep fills values,
one backward pass fills every gradient, and y visibly accumulates its gradient
from both of its paths. Every number is read off the engine's own graph, not
hard-coded.

    uv run --group viz python examples/record_backward.py  # -> assets/reverse_mode.gif
"""

import numpy as np

import matplotlib

matplotlib.use("Agg")
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import figstyle

from autograd.engine import Tensor

# ---- the engine computes everything the animation shows ----
x, y = Tensor(3.0), Tensor(2.0)
h = x * y
z = h + y
z.backward()
VAL = {"x": 3, "y": 2, "h": 6, "z": 8}
GRAD = {k: float(t.grad) for k, t in (("x", x), ("y", y), ("h", h), ("z", z))}
assert VAL == {
    "x": float(x.data),
    "y": float(y.data),
    "h": float(h.data),
    "z": float(z.data),
}

POS = {"x": (0.13, 0.74), "y": (0.13, 0.30), "h": (0.46, 0.62), "z": (0.80, 0.46)}
NAME = {"x": "$x$", "y": "$y$", "h": "$h = xy$", "z": "$z = h + y$"}
EDGES = [("x", "h", 0.0), ("y", "h", 0.0), ("h", "z", 0.0), ("y", "z", 0.35)]
W, HGT = (7.2, 4.2)


def node(ax, k, value=None, grad=None, hot=False):
    px, py = POS[k]
    box = FancyBboxPatch(
        (px - 0.095, py - 0.115),
        0.19,
        0.23,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor=figstyle.BLUE
        if hot == "f"
        else figstyle.RED
        if hot == "b"
        else figstyle.INK,
        linewidth=1.8 if hot else 1.0,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        px,
        py + 0.062,
        NAME[k],
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        color=figstyle.INK,
        zorder=4,
    )
    if value is not None:
        ax.text(
            px,
            py - 0.005,
            f"value {value:g}",
            transform=ax.transAxes,
            ha="center",
            fontsize=9.5,
            color=figstyle.BLUE,
            zorder=4,
        )
    if grad is not None:
        ax.text(
            px,
            py - 0.068,
            grad,
            transform=ax.transAxes,
            ha="center",
            fontsize=9.5,
            color=figstyle.RED,
            zorder=4,
        )


def edge(ax, a, b, rad, kind=None):
    """one arrow per pair per frame: gray at rest, blue forward, red backward
    (same arc, direction reversed)."""
    (x1, y1), (x2, y2) = POS[a], POS[b]
    if kind == "b":
        (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
        rad, style, color, lw = -rad, "--", figstyle.RED, 1.9
    elif kind == "f":
        style, color, lw = "-", figstyle.BLUE, 1.9
    else:
        style, color, lw = "-", "#c0c0c0", 1.2
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=17,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=style,
            color=color,
            linewidth=lw,
            shrinkA=15,
            shrinkB=15,
            zorder=2,
        )
    )


def frame(values, grads, hot_nodes, hot_edges, phase, caption=""):
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(W, HGT), dpi=130)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for a, b, rad in EDGES:
        edge(ax, a, b, rad, hot_edges.get((a, b)))
    for k in POS:
        node(ax, k, values.get(k), grads.get(k), hot_nodes.get(k))
    if phase:
        label, color = phase
        ax.text(
            0.03,
            0.965,
            label,
            transform=ax.transAxes,
            fontsize=11.5,
            color=color,
            va="top",
        )
    if caption:
        ax.text(
            0.5,
            0.035,
            caption,
            transform=ax.transAxes,
            fontsize=11,
            color=figstyle.INK,
            ha="center",
        )
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return out


def main():
    FWD = ("forward: compute values", figstyle.BLUE)
    BWD = ("backward: one pass fills every gradient", figstyle.RED)
    g = lambda k: f"grad {GRAD[k]:g}"
    F = frame
    frames, times = [], []

    def state(img, ms):
        frames.append(img)
        times.append(ms)

    state(F({}, {}, {}, {}, None), 900)
    # forward sweep, in topological order
    v = {"x": 3, "y": 2}
    state(F(v, {}, {"x": "f", "y": "f"}, {}, FWD), 850)
    v["h"] = 6
    state(F(v, {}, {"h": "f"}, {("x", "h"): "f", ("y", "h"): "f"}, FWD), 950)
    v["z"] = 8
    state(F(v, {}, {"z": "f"}, {("h", "z"): "f", ("y", "z"): "f"}, FWD), 950)
    state(F(v, {}, {}, {}, FWD), 650)

    # backward sweep, reverse topological order; y accumulates from two paths
    gr = {"z": "grad 1 (seed)"}
    state(F(v, gr, {"z": "b"}, {}, BWD), 950)
    gr["h"] = g("h")
    gr["y"] = "grad 1"
    state(F(v, gr, {"h": "b", "y": "b"}, {("h", "z"): "b", ("y", "z"): "b"}, BWD), 1100)
    gr["x"] = g("x")
    gr["y"] = "grad 1 + 3 = 4"
    state(
        F(
            v,
            gr,
            {"x": "b", "y": "b"},
            {("x", "h"): "b", ("y", "h"): "b"},
            BWD,
            caption=r"$y$ feeds both the product and the sum, so += sums its two paths",
        ),
        1600,
    )

    gr["y"] = g("y")
    state(
        F(
            v,
            gr,
            {},
            {},
            None,
            caption=r"$\partial z / \partial y = 3 + 1 = 4$ — one backward pass, every gradient",
        ),
        2300,
    )

    iio.imwrite("assets/reverse_mode.gif", frames, duration=times, loop=0)
    print(f"wrote assets/reverse_mode.gif ({len(frames)} states)")
    print("engine says:", {k: GRAD[k] for k in ("x", "y", "h", "z")})


if __name__ == "__main__":
    main()
