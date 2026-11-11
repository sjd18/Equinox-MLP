"""Timelapse: watch the 2-D cluster's physics improve as the MLP learns.

Snapshots of the model are recorded during training (via ``train``'s
``callback``). For every snapshot, the cluster is simulated from the *same*
starting configuration using that snapshot's learned forces ``-∇V_θ``. Each
GIF frame shows

* left:   where the atoms are after 10 time units of MD with those forces,
* middle: the learned pair potential ``φ_θ(r)`` against the true one,
* right:  the training loss so far.

Saves ``assets/training_timelapse.gif``.

    python examples/training_timelapse.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import _style  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from common import ASSETS, DT, PAIR_HPARAMS, TRAIN, datasets  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import eqxmlp as em  # noqa: E402

SIM_STEPS = 2000  # 10 time units of MD per snapshot
SAVE_EVERY = 10
TRAIL = 30  # saved frames of motion trail (1.5 time units)
VIEW = 4.0  # half-width of the cluster panel


def snapshot_steps(total: int, n: int = 70) -> set[int]:
    """Dense early (where the loss falls fastest), sparse late."""
    return {0, *np.unique(np.geomspace(1, total, n).round().astype(int)).tolist()}


def main():
    train_data, test_data = datasets()
    keep = snapshot_steps(TRAIN["steps"])
    snapshots = []
    model = em.PairPotentialMLP(**PAIR_HPARAMS, key=jax.random.PRNGKey(1))
    print(f"training with {len(keep)} snapshots...")
    _, losses = em.train(
        model,
        train_data,
        key=jax.random.PRNGKey(2),
        **TRAIN,
        verbose=False,
        callback=lambda n, m: snapshots.append((n, m)) if n in keep else None,
    )
    losses = np.asarray(losses)

    lj = em.LennardJones()
    start = em.initial_state(jax.random.PRNGKey(7))
    r_grid = jnp.linspace(0.88, 2.7, 300)
    phi_true = lj.pair(r_grid)

    print("simulating each snapshot...")
    frames = []
    for n, m in snapshots:
        tr = em.simulate(m.forces, start, dt=DT, n_steps=SIM_STEPS, save_every=SAVE_EVERY)
        q = np.asarray(tr.q)
        r2 = em.evaluate(m, test_data)["force_r2"]
        frames.append((n, q, np.asarray(jax.vmap(m.pair)(r_grid)), r2))
    truth = np.asarray(
        em.simulate(lambda x: em.forces(lj, x), start, dt=DT, n_steps=SIM_STEPS, save_every=SAVE_EVERY).q
    )

    fig, (ax_c, ax_p, ax_l) = plt.subplots(
        1, 3, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1, 1.15, 1.15]}
    )
    n_atoms = start.q.shape[0]

    # Cluster panel.
    ax_c.set_xlim(-VIEW, VIEW)
    ax_c.set_ylim(-VIEW, VIEW)
    ax_c.set_aspect("equal")
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.grid(False)
    for side in ["left", "bottom", "top", "right"]:
        ax_c.spines[side].set_visible(True)
        ax_c.spines[side].set_color(_style.GRID)
    ghost = ax_c.scatter(
        truth[-1, :, 0], truth[-1, :, 1], s=150, facecolor="none", edgecolor=_style.GRID, lw=1.5
    )
    trails = [ax_c.plot([], [], color=_style.BLUE, lw=0.8, alpha=0.35)[0] for _ in range(n_atoms)]
    atoms = ax_c.scatter([], [], s=110, color=_style.BLUE, edgecolor=_style.SURFACE, lw=1.5, zorder=3)
    ax_c.set_title("Cluster under learned forces")
    ax_c.legend(
        [atoms, ghost],
        ["learned forces", "true forces"],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.13),
        fontsize=9,
    )

    # Potential panel.
    ax_p.plot(r_grid, phi_true, color=_style.TEXT, ls="--", lw=1.2, label="true Lennard-Jones", zorder=3)
    (phi_line,) = ax_p.plot([], [], color=_style.BLUE, lw=3, label="learned φ_θ(r)")
    ax_p.set_xlim(0.88, 2.7)
    ax_p.set_ylim(-1.3, 1.5)
    ax_p.set_xlabel("pair distance r")
    ax_p.set_title("Learned pair potential")
    ax_p.legend(loc="upper right", fontsize=9)

    # Loss panel.
    steps_axis = np.arange(1, len(losses) + 1)
    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xlim(1, len(losses))
    ax_l.set_ylim(losses.min() * 0.5, losses.max() * 2)
    (loss_line,) = ax_l.plot([], [], color=_style.BLUE, lw=1)
    (loss_dot,) = ax_l.plot([], [], "o", color=_style.BLUE, ms=8, mec=_style.SURFACE, mew=2)
    ax_l.set_xlabel("optimisation step")
    ax_l.set_title("Training loss")

    header = fig.suptitle("", x=0.01, ha="left", fontweight="bold", color=_style.TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def draw(k):
        n, q, phi, r2 = frames[k]
        for i, line in enumerate(trails):
            line.set_data(q[-TRAIL:, i, 0], q[-TRAIL:, i, 1])
        atoms.set_offsets(q[-1])
        phi_line.set_data(r_grid, np.clip(phi, -10, 10))
        loss_line.set_data(steps_axis[: max(n, 1)], losses[: max(n, 1)])
        loss_dot.set_data([max(n, 1)], [losses[max(n, 1) - 1]])
        r2_text = f"{r2:.5f}" if r2 > 0.99 else f"{r2:.3f}" if r2 > -9.99 else "< -10"
        header.set_text(f"Training step {n:>4d} / {len(losses)}    held-out force R² = {r2_text}")
        return [*trails, atoms, phi_line, loss_line, loss_dot, header]

    # Hold the first and last frames a little longer.
    order = [0] * 6 + list(range(len(frames))) + [len(frames) - 1] * 20
    anim = FuncAnimation(fig, draw, frames=order, blit=False)
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / "training_timelapse.gif"
    anim.save(out, writer=PillowWriter(fps=8), dpi=100)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB, {len(order)} frames)")


if __name__ == "__main__":
    main()
