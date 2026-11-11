"""Animate training: the cluster's physics improving as the MLP learns.

Typical use::

    keep = em.snapshot_steps(3000)
    snapshots = []
    model, losses = em.train(
        model, data, key=key, steps=3000,
        callback=lambda n, m: snapshots.append((n, m)) if n in keep else None,
    )
    em.render_timelapse(snapshots, losses, test_data=test, start=start, path="timelapse.gif")

Each frame simulates the cluster from ``start`` with one snapshot's learned
forces and shows it next to the learned pair potential and the loss so far.
Requires matplotlib (``pip install -e ".[examples]"``).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .integrators import simulate
from .systems import LennardJones, PhaseSpace, forces
from .training import evaluate

__all__ = ["snapshot_steps", "render_timelapse"]

_SURFACE, _TEXT, _TEXT_2, _GRID, _BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
_RC = {
    "figure.facecolor": _SURFACE,
    "axes.facecolor": _SURFACE,
    "savefig.facecolor": _SURFACE,
    "axes.edgecolor": _GRID,
    "axes.labelcolor": _TEXT_2,
    "axes.titlecolor": _TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": _GRID,
    "grid.linewidth": 0.8,
    "xtick.color": _TEXT_2,
    "ytick.color": _TEXT_2,
    "legend.frameon": False,
    "font.size": 10,
}


def snapshot_steps(total: int, n: int = 70) -> set[int]:
    """Training steps at which to keep a snapshot: step 0 plus ``n``
    log-spaced steps, dense early (where the loss falls fastest), sparse late."""
    return {0, *np.unique(np.geomspace(1, total, n).round().astype(int)).tolist()}


def render_timelapse(
    snapshots,
    losses,
    *,
    test_data,
    start: PhaseSpace,
    path,
    dt: float = 0.005,
    sim_steps: int = 2000,
    save_every: int = 10,
    trail: int = 30,
    view: float = 4.0,
    fps: int = 8,
    dpi: int = 100,
    potential=LennardJones(),
) -> Path:
    """Render a GIF of training snapshots and return its path.

    Args:
        snapshots: list of ``(step, model)`` pairs, e.g. collected with
            ``train(..., callback=...)``. Models need ``forces`` and ``pair``.
        losses: the per-step loss array returned by ``train``.
        test_data: held-out :class:`~eqxmlp.Dataset`, for the R² in each frame.
        start: initial condition simulated with every snapshot's forces.
        path: where to write the GIF.
        sim_steps, dt, save_every: MD length, step and sampling per snapshot.
        trail: number of saved frames drawn as a motion trail.
        potential: the true potential, drawn as hollow "ghost" atoms and as
            the reference curve.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    path = Path(path)
    losses = np.asarray(losses)
    r_grid = jnp.linspace(0.88, 2.7, 300)

    frames = []
    for n, m in sorted(snapshots, key=lambda s: s[0]):
        q = np.asarray(simulate(m.forces, start, dt=dt, n_steps=sim_steps, save_every=save_every).q)
        frames.append((n, q, np.asarray(jax.vmap(m.pair)(r_grid)), evaluate(m, test_data)["force_r2"]))
    truth = np.asarray(
        simulate(lambda x: forces(potential, x), start, dt=dt, n_steps=sim_steps, save_every=save_every).q
    )

    with plt.rc_context(_RC):
        fig, (ax_c, ax_p, ax_l) = plt.subplots(
            1, 3, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1, 1.15, 1.15]}
        )

        ax_c.set_xlim(-view, view)
        ax_c.set_ylim(-view, view)
        ax_c.set_aspect("equal")
        ax_c.set_xticks([])
        ax_c.set_yticks([])
        ax_c.grid(False)
        for side in ["left", "bottom", "top", "right"]:
            ax_c.spines[side].set_visible(True)
        ghost = ax_c.scatter(
            truth[-1, :, 0], truth[-1, :, 1], s=150, facecolor="none", edgecolor=_GRID, lw=1.5
        )
        trails = [ax_c.plot([], [], color=_BLUE, lw=0.8, alpha=0.35)[0] for _ in range(start.q.shape[0])]
        atoms = ax_c.scatter([], [], s=110, color=_BLUE, edgecolor=_SURFACE, lw=1.5, zorder=3)
        ax_c.set_title("Cluster under learned forces")
        ax_c.legend(
            [atoms, ghost],
            ["learned forces", "true forces"],
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, -0.13),
            fontsize=9,
        )

        ax_p.plot(r_grid, potential.pair(r_grid), color=_TEXT, ls="--", lw=1.2, label="true", zorder=3)
        (phi_line,) = ax_p.plot([], [], color=_BLUE, lw=3, label="learned φ_θ(r)")
        ax_p.set_xlim(0.88, 2.7)
        ax_p.set_ylim(-1.3, 1.5)
        ax_p.set_xlabel("pair distance r")
        ax_p.set_title("Learned pair potential")
        ax_p.legend(loc="upper right", fontsize=9)

        steps_axis = np.arange(1, len(losses) + 1)
        ax_l.set_xscale("log")
        ax_l.set_yscale("log")
        ax_l.set_xlim(1, len(losses))
        ax_l.set_ylim(losses.min() * 0.5, losses.max() * 2)
        (loss_line,) = ax_l.plot([], [], color=_BLUE, lw=1)
        (loss_dot,) = ax_l.plot([], [], "o", color=_BLUE, ms=8, mec=_SURFACE, mew=2)
        ax_l.set_xlabel("optimisation step")
        ax_l.set_title("Training loss")

        header = fig.suptitle("", x=0.01, ha="left", fontweight="bold", color=_TEXT)
        fig.tight_layout(rect=(0, 0, 1, 0.93))

        def draw(k):
            n, q, phi, r2 = frames[k]
            for i, line in enumerate(trails):
                line.set_data(q[-trail:, i, 0], q[-trail:, i, 1])
            atoms.set_offsets(q[-1])
            phi_line.set_data(r_grid, np.clip(phi, -10, 10))
            shown = max(n, 1)
            loss_line.set_data(steps_axis[:shown], losses[:shown])
            loss_dot.set_data([shown], [losses[shown - 1]])
            r2_text = f"{r2:.5f}" if r2 > 0.99 else f"{r2:.3f}" if r2 > -9.99 else "< -10"
            header.set_text(f"Training step {n:>4d} / {len(losses)}    held-out force R² = {r2_text}")
            return [*trails, atoms, phi_line, loss_line, loss_dot, header]

        # Hold the first and last frames a little longer.
        order = [0] * 6 + list(range(len(frames))) + [len(frames) - 1] * 20
        anim = FuncAnimation(fig, draw, frames=order, blit=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
    return path
