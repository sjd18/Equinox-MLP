"""Train an Equinox MLP pair potential on Lennard-Jones forces and energies.

    python examples/train_pair_potential.py

Saves the model to ``checkpoints/pair_potential.eqx`` and the figure
``assets/learned_potential.png``.
"""

import time

import jax

jax.config.update("jax_enable_x64", True)

import _style  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from common import ASSETS, PAIR_HPARAMS, TRAIN, datasets, save_pair_model  # noqa: E402

import eqxmlp as em  # noqa: E402


def main():
    train_data, test_data = datasets()
    print(
        f"training frames: {len(train_data)}   test frames: {len(test_data)}   atoms: {train_data.q.shape[1]}"
    )

    model = em.PairPotentialMLP(**PAIR_HPARAMS, key=jax.random.PRNGKey(1))
    n_params = sum(x.size for x in jax.tree.leaves(model) if hasattr(x, "size"))
    print(f"PairPotentialMLP with {n_params} parameters")

    t0 = time.perf_counter()
    model, losses = em.train(model, train_data, key=jax.random.PRNGKey(2), **TRAIN)
    print(f"trained in {time.perf_counter() - t0:.0f} s")
    metrics = em.evaluate(model, test_data)
    print("held-out test set:", {k: f"{v:.2e}" if v < 0.1 else f"{v:.6f}" for k, v in metrics.items()})
    save_pair_model(model)

    # Range of distances actually present in the training data.
    i, j = jnp.triu_indices(train_data.q.shape[1], k=1)
    r_train = jnp.linalg.norm(train_data.q[:, i] - train_data.q[:, j], axis=-1)
    r_min = float(r_train.min())

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.2, 1, 1]})

    ax = axes[0]
    r = jnp.linspace(0.88, 2.7, 400)
    ax.axvspan(0.88, r_min, color=_style.GRID, lw=0)
    ax.text(r_min + 0.03, 1.4, "← closest pair seen in training", fontsize=8, color=_style.TEXT_2, va="top")
    ax.plot(r, jax.vmap(model.pair)(r), color=_style.BLUE, lw=3, label="Learned φ_θ(r)")
    ax.plot(r, em.LennardJones().pair(r), color=_style.TEXT, ls="--", lw=1.2, label="True Lennard-Jones")
    ax.set_ylim(-1.25, 1.5)
    ax.set_xlabel("pair distance r")
    ax.set_ylabel("pair energy")
    ax.set_title("Learned pair potential")
    ax.legend(loc="lower right")

    ax = axes[1]
    f_true = test_data.forces.ravel()
    f_pred = jax.vmap(model.forces)(test_data.q).ravel()
    lim = float(jnp.abs(f_true).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=_style.TEXT_2, lw=1, ls="--")
    ax.scatter(f_true, f_pred, s=2, color=_style.BLUE, alpha=0.25, lw=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("true force component")
    ax.set_ylabel("predicted  −∂V_θ/∂q")
    ax.set_title(f"Held-out forces, R² = {metrics['force_r2']:.5f}")

    ax = axes[2]
    ax.semilogy(losses, color=_style.BLUE, lw=1)
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("energy + force loss")
    ax.set_title("Training")

    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "learned_potential.png")
    print("saved", ASSETS / "learned_potential.png")


if __name__ == "__main__":
    main()
