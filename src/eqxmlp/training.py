"""Force-matching training with Equinox and Optax.

For a potential model the loss compares *both* the predicted energy and the
predicted forces with the labels. The predicted forces are themselves
``-jax.grad`` of the network, so training differentiates through a gradient:
``eqx.filter_value_and_grad`` of a loss that contains ``jax.grad``. JAX
composes the two transformations without any special handling.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray

from .data import Dataset

__all__ = ["loss_fn", "train", "evaluate"]


def loss_fn(model, batch: Dataset, energy_weight: float = 1.0, force_weight: float = 1.0) -> Array:
    """Per-atom energy MSE + per-component force MSE.

    Models without an ``energy`` method (``DirectForceMLP``) are trained on
    forces only.
    """
    n_atoms = batch.q.shape[1]
    f_pred = jax.vmap(model.forces)(batch.q)
    loss = force_weight * jnp.mean((f_pred - batch.forces) ** 2)
    if hasattr(model, "energy"):
        e_pred = jax.vmap(model.energy)(batch.q)
        # Energies are only defined up to a constant offset, which the MLP's
        # bias learns; compare per-atom so the scale matches the forces.
        loss = loss + energy_weight * jnp.mean(((e_pred - batch.energy) / n_atoms) ** 2)
    return loss


def evaluate(model, data: Dataset) -> dict[str, float]:
    """Force RMSE, force R² and (if available) per-atom energy RMSE."""
    f_pred = jax.vmap(model.forces)(data.q)
    resid = f_pred - data.forces
    out = {
        "force_rmse": float(jnp.sqrt(jnp.mean(resid**2))),
        "force_r2": float(1.0 - jnp.sum(resid**2) / jnp.sum((data.forces - data.forces.mean()) ** 2)),
    }
    if hasattr(model, "energy"):
        e_pred = jax.vmap(model.energy)(data.q)
        out["energy_rmse_per_atom"] = float(
            jnp.sqrt(jnp.mean(((e_pred - data.energy) / data.q.shape[1]) ** 2))
        )
    return out


def train(
    model,
    data: Dataset,
    *,
    key: PRNGKeyArray,
    steps: int = 3000,
    batch_size: int = 128,
    lr: float = 3e-3,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    log_every: int = 500,
    verbose: bool = True,
    callback=None,
):
    """Adam with cosine decay over random minibatches. Returns ``(model, losses)``.

    If given, ``callback(n, model)`` is called with the model after ``n``
    optimisation steps, for ``n = 0`` (untrained) through ``steps``. Use it to
    record snapshots, e.g. for ``examples/training_timelapse.py``.
    """
    schedule = optax.cosine_decay_schedule(lr, steps, alpha=0.01)
    opt = optax.adam(schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model, opt_state, idx):
        batch = jax.tree.map(lambda x: x[idx], data)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch, energy_weight, force_weight)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, updates), opt_state, loss

    losses = []
    if callback is not None:
        callback(0, model)
    for i in range(steps):
        key, sub = jax.random.split(key)
        idx = jax.random.choice(sub, len(data), (batch_size,), replace=False)
        model, opt_state, loss = step(model, opt_state, idx)
        losses.append(float(loss))
        if callback is not None:
            callback(i + 1, model)
        if verbose and (i % log_every == 0 or i == steps - 1):
            print(f"  step {i:5d}  loss {float(loss):.3e}")
    return model, jnp.array(losses)
