"""Why learn a *symmetric potential*? Compare against two unstructured MLPs.

* Pair-potential MLP (this repo): energy from pair distances; forces = −∇V_θ.
* Global-potential MLP: energy from raw coordinates; forces = −∇V_θ.
  Conservative, but not translation/rotation/permutation invariant.
* Direct-force MLP: forces straight from raw coordinates. No energy at all.

All three are integrated with the same velocity-Verlet + lax.scan code.
Saves ``assets/baselines.png``.

    python examples/baselines.py
"""

import time

import jax

jax.config.update("jax_enable_x64", True)

import _style  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from common import ASSETS, DT, datasets, load_pair_model  # noqa: E402

import eqxmlp as em  # noqa: E402

BASELINE_TRAIN = {"steps": 20_000, "batch_size": 128, "lr": 1e-3}


def n_params(m):
    return sum(x.size for x in jax.tree.leaves(m) if hasattr(x, "size"))


def main(n_steps: int = 20_000, save_every: int = 50):
    train_data, test_data = datasets()
    n_atoms = train_data.q.shape[1]

    pair = load_pair_model()
    if pair is None:
        import train_pair_potential

        train_pair_potential.main()
        pair = load_pair_model()

    models = {"Pair-potential MLP": pair}
    for name, m in [
        ("Global-potential MLP", em.GlobalPotentialMLP(n_atoms, key=jax.random.PRNGKey(3))),
        ("Direct-force MLP", em.DirectForceMLP(n_atoms, key=jax.random.PRNGKey(4))),
    ]:
        print(f"training {name} ({BASELINE_TRAIN['steps']} steps)...")
        t0 = time.perf_counter()
        models[name], _ = em.train(m, train_data, key=jax.random.PRNGKey(2), verbose=False, **BASELINE_TRAIN)
        print(f"  done in {time.perf_counter() - t0:.0f} s")

    lj = em.LennardJones()
    s0 = em.initial_state(jax.random.PRNGKey(7))
    results = {}
    print(f"\n{'model':22s} {'params':>7s} {'test force R²':>14s} {'max |ΔE_true/E|':>16s} {'max |P|':>9s}")
    for name, m in models.items():
        r2 = em.evaluate(m, test_data)["force_r2"]
        tr = em.simulate(m.forces, s0, dt=DT, n_steps=n_steps, save_every=save_every)
        E = em.total_energy(lj, tr)
        dE = jnp.abs(E - E[0]) / jnp.abs(E[0])
        P = jnp.linalg.norm(em.total_momentum(tr.p), axis=-1)
        own = ""
        if hasattr(m, "energy"):
            Eo = em.total_energy(m.energy, tr)
            own = f"   (own-energy drift {float(jnp.abs(Eo - Eo[0]).max() / jnp.abs(Eo[0])):.1e})"
        results[name] = (tr.t, dE, P)
        print(f"{name:22s} {n_params(m):7d} {r2:14.5f} {float(dE.max()):16.2e} {float(P.max()):9.1e}{own}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    for name, (t, dE, P) in results.items():
        c = _style.COLORS[name]
        axes[0].semilogy(t[1:], jnp.maximum(dE[1:], 1e-12), color=c, label=name)
        axes[1].semilogy(t[1:], jnp.maximum(P[1:], 1e-16), color=c, label=name)
    axes[0].set_title("True energy along each learned trajectory")
    axes[0].set_ylabel("|E_true(t) − E_true(0)| / |E_true(0)|")
    axes[1].set_title("Total momentum (starts at exactly 0)")
    axes[1].set_ylabel("|P(t)|")
    for ax in axes:
        ax.set_xlabel("time")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "baselines.png")
    print("saved", ASSETS / "baselines.png")


if __name__ == "__main__":
    main()
