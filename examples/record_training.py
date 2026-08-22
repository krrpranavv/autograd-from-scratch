"""Render the README gif: the spiral MLP's decision boundary taking shape as it
trains, every gradient computed by this engine. Reruns exactly the training in
train_mlp.py (same seeds, same consumption order) and snapshots the boundary
along the way.

    uv run --group viz python examples/record_training.py   # -> assets/spiral.gif
"""

import numpy as np

import matplotlib

matplotlib.use("Agg")
import imageio.v3 as iio
import matplotlib.pyplot as plt

import figstyle
from train_mlp import MLP, make_spiral

from autograd.engine import Tensor, cross_entropy
from autograd.nn import Adam

STEPS = 400
GRID = 220


def snapshot_steps():
    """every step while the boundary moves fast, sparser as it sharpens."""
    steps = (
        list(range(0, 40)) + list(range(40, 120, 3)) + list(range(120, STEPS + 1, 12))
    )
    return sorted(set(steps))


def render(model, X, y, step, loss, xx, yy, grid):
    logits = model(Tensor(grid)).data
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = (p / p.sum(axis=1, keepdims=True))[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(4.6, 4.3), dpi=130)
    ax.contourf(
        xx, yy, p, levels=np.linspace(0, 1, 12), cmap="RdBu", alpha=0.55, zorder=1
    )
    ax.contour(xx, yy, p, levels=[0.5], colors=figstyle.INK, linewidths=1.1, zorder=2)
    for c, col in ((0, figstyle.RED), (1, figstyle.BLUE)):
        m = y == c
        ax.scatter(
            X[m, 0],
            X[m, 1],
            s=9,
            color=col,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )
    acc = (model(Tensor(X)).data.argmax(axis=1) == y).mean()
    ax.set_title(
        f"step {step:3d}   loss {loss:.3f}   acc {acc:.2f}",
        fontsize=10.5,
        family="monospace",
    )
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(figstyle.LIGHT)
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def main():
    figstyle.apply()
    # identical seeding + consumption order to train_mlp.train()
    np.random.seed(0)
    X, y = make_spiral(n_per_class=100, classes=2)
    x = Tensor(X)
    model = MLP(2, 32, 2)
    opt = Adam(model.parameters(), lr=0.05)

    lin = np.linspace(-1.15, 1.15, GRID)
    xx, yy = np.meshgrid(lin, lin)
    grid = np.c_[xx.ravel(), yy.ravel()]

    snaps, frames = set(snapshot_steps()), []
    loss_val = float("nan")
    for step in range(STEPS + 1):
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss_val = float(loss.data)
        if step in snaps:
            frames.append(render(model, X, y, step, loss_val, xx, yy, grid))
        if step == STEPS:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()

    frames += [frames[-1]] * 14  # hold the finished boundary
    iio.imwrite("assets/spiral.gif", frames, duration=70, loop=0)
    print(f"wrote assets/spiral.gif ({len(frames)} frames, final loss {loss_val:.3f})")


if __name__ == "__main__":
    main()
