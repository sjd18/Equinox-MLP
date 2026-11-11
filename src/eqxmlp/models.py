"""Equinox models that learn interatomic physics.

``PairPotentialMLP`` is the main model: an ``eqx.nn.MLP`` maps a distance
``r`` to a pair energy ``φ_θ(r)``, summed over all pairs. Its forces are
``-∇V_θ`` from ``jax.grad``. Learning a *potential* makes the forces
conservative (energy is conserved). Building it from *pair distances* makes
it invariant to translation, rotation and atom relabelling, so momentum and
angular momentum are conserved too, exactly and by construction.

Two baselines each drop one of those ingredients:

* ``GlobalPotentialMLP`` still learns an energy (conservative) but reads the
  raw coordinates, so it has no symmetry and momentum is not conserved.
* ``DirectForceMLP`` predicts forces directly, with no energy behind them, so
  nothing stops energy from drifting.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from .systems import pairwise_sum, smooth_cutoff

__all__ = ["PairPotentialMLP", "GlobalPotentialMLP", "DirectForceMLP"]


class PairPotentialMLP(eqx.Module):
    """``V_θ(q) = Σ_{i<j} φ_θ(r_ij)``, ``φ_θ(r) = MLP([r, 1/r]) · S(r)``.

    The ``1/r`` feature makes the steep short-range wall easy to fit; the
    switching function ``S`` gives the learned potential the same smooth
    cutoff as the ground truth.
    """

    mlp: eqx.nn.MLP
    r_on: float = eqx.field(static=True)
    r_cut: float = eqx.field(static=True)

    def __init__(
        self, width: int = 64, depth: int = 3, r_on: float = 2.0, r_cut: float = 2.5, *, key: PRNGKeyArray
    ):
        self.mlp = eqx.nn.MLP(2, "scalar", width, depth, activation=jax.nn.silu, key=key)
        self.r_on, self.r_cut = r_on, r_cut

    def pair(self, r: Array) -> Array:
        """The learned pair energy ``φ_θ(r)`` for a scalar distance."""
        return self.mlp(jnp.stack([r, 1.0 / r])) * smooth_cutoff(r, self.r_on, self.r_cut)

    def energy(self, q: Array) -> Array:
        return pairwise_sum(self.pair, q)

    def forces(self, q: Array) -> Array:
        return -jax.grad(self.energy)(q)


class GlobalPotentialMLP(eqx.Module):
    """``V_θ(q) = MLP(flatten(q))``. Conservative, but no symmetry."""

    mlp: eqx.nn.MLP

    def __init__(self, n_atoms: int, dim: int = 2, width: int = 128, depth: int = 3, *, key: PRNGKeyArray):
        self.mlp = eqx.nn.MLP(n_atoms * dim, "scalar", width, depth, activation=jax.nn.silu, key=key)

    def energy(self, q: Array) -> Array:
        return self.mlp(q.ravel())

    def forces(self, q: Array) -> Array:
        return -jax.grad(self.energy)(q)


class DirectForceMLP(eqx.Module):
    """``F_θ(q) = MLP(flatten(q))``. Predicts forces with no energy behind them."""

    mlp: eqx.nn.MLP

    def __init__(self, n_atoms: int, dim: int = 2, width: int = 128, depth: int = 3, *, key: PRNGKeyArray):
        self.mlp = eqx.nn.MLP(n_atoms * dim, n_atoms * dim, width, depth, activation=jax.nn.silu, key=key)

    def forces(self, q: Array) -> Array:
        return self.mlp(q.ravel()).reshape(q.shape)
