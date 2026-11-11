"""Timelapse: watch the 2-D cluster's physics improve as the MLP learns.

Snapshots of the model are recorded during training (via ``train``'s
``callback``). For every snapshot, the cluster is simulated from the *same*
starting configuration using that snapshot's learned forces ``-∇V_θ``.
Rendering is done by :func:`eqxmlp.render_timelapse`. Saves
``assets/training_timelapse.gif``.

    python examples/training_timelapse.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import _style  # noqa: E402, F401  (sets the non-interactive backend)
from common import ASSETS, DT, PAIR_HPARAMS, TRAIN, datasets  # noqa: E402

import eqxmlp as em  # noqa: E402


def main():
    train_data, test_data = datasets()
    keep = em.snapshot_steps(TRAIN["steps"])
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
    print("simulating each snapshot and rendering...")
    out = em.render_timelapse(
        snapshots,
        losses,
        test_data=test_data,
        start=em.initial_state(jax.random.PRNGKey(7)),
        path=ASSETS / "training_timelapse.gif",
        dt=DT,
    )
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
