"""Molecular dynamics driven by the learned potential's autodiff forces.

Runs velocity Verlet from the same initial condition twice, once with the
true Lennard-Jones forces and once with ``-∇V_θ`` from the trained Equinox
MLP, and compares the physics. Saves ``assets/md_learned.png``.

    python examples/simulate_learned.py      # trains first if no checkpoint exists
"""

import time

import jax

jax.config.update("jax_enable_x64", True)

import _style  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from common import ASSETS, DT, load_pair_model  # noqa: E402

import eqxmlp as em  # noqa: E402


def main(n_steps: int = 100_000, save_every: int = 50):
    model = load_pair_model()
    if model is None:
        import train_pair_potential

        train_pair_potential.main()
        model = load_pair_model()

    lj = em.LennardJones()
    s0 = em.initial_state(jax.random.PRNGKey(7))  # not used for training
    runs = {}
    for name, potential in [("True Lennard-Jones", lj), ("Learned MLP potential", model.energy)]:
        force_fn = lambda q, V=potential: em.forces(V, q)  # noqa: E731
        em.simulate(force_fn, s0, dt=DT, n_steps=save_every, save_every=save_every)  # compile
        t0 = time.perf_counter()
        tr = em.simulate(force_fn, s0, dt=DT, n_steps=n_steps, save_every=save_every)
        tr.q.block_until_ready()
        dt_wall = time.perf_counter() - t0
        E = em.total_energy(potential, tr)
        runs[name] = (tr, E)
        P = jnp.abs(em.total_momentum(tr.p)).max()
        L = jnp.abs(em.angular_momentum(tr.q, tr.p)).max()
        dE = float(jnp.abs(E - E[0]).max() / abs(E[0]))
        print(
            f"{name:22s} {n_steps} steps in {dt_wall:.1f} s | max |ΔE/E| {dE:.1e}"
            f" | max |P| {float(P):.1e} | max |L| {float(L):.1e}"
        )

    (tr_true, E_true), (tr_ml, E_ml) = runs.values()
    sep = jnp.sqrt(jnp.mean((tr_true.q - tr_ml.q) ** 2, axis=(1, 2)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    colors = {"True Lennard-Jones": _style.TEXT_2, "Learned MLP potential": _style.BLUE}

    ax = axes[0]
    for name, (tr, E) in runs.items():
        ax.plot(tr.t, (E - E[0]) / jnp.abs(E[0]), color=colors[name], lw=1, label=name)
    ax.set_xlabel("time")
    ax.set_ylabel("(E(t) − E(0)) / |E(0)|")
    ax.set_title("Energy conservation (each w.r.t. its own V)")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    for name, (tr, _) in runs.items():
        r, dens = em.radial_distribution(tr.q[len(tr.q) // 5 :], r_max=3.5, n_bins=140)
        ls = "--" if name.startswith("True") else "-"
        lw = 1.5 if name.startswith("True") else 2.5
        ax.plot(r, dens, color=colors[name], ls=ls, lw=lw, label=name, zorder=3 if ls == "--" else 2)
    ax.set_xlabel("pair distance r")
    ax.set_ylabel("density of pairs")
    ax.set_title("Structure: pair-distance distribution")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[2]
    ax.semilogy(tr_true.t[1:], sep[1:], color=_style.BLUE, lw=1.5)
    ax.set_xlabel("time")
    ax.set_ylabel("RMS position difference")
    ax.set_title("Trajectories: learned vs true")
    ax.text(
        0.97,
        0.05,
        "chaos amplifies tiny force errors;\nstatistics (middle) still agree",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=_style.TEXT_2,
    )

    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "md_learned.png")
    print("saved", ASSETS / "md_learned.png")


if __name__ == "__main__":
    main()
