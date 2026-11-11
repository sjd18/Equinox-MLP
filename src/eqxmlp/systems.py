"""Ground-truth physics: a Lennard-Jones cluster, plus shared pair geometry.

Everything here is a pure function of positions ``q`` with shape ``(N, dim)``.
Forces are never written by hand: they are ``-jax.grad`` of an energy.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

__all__ = [
    "PhaseSpace",
    "pair_distances",
    "pairwise_sum",
    "smooth_cutoff",
    "lennard_jones",
    "LennardJones",
    "forces",
    "hexagonal_cluster",
    "maxwell_boltzmann",
    "remove_rotation",
]


class PhaseSpace(NamedTuple):
    """Positions and momenta, each ``(N, dim)``. A NamedTuple is a pytree."""

    q: Array
    p: Array


def pair_distances(q: Array) -> tuple[Array, Array]:
    """All pairwise distances ``r_ij`` and the self-pair mask.

    Uses the "double-where" trick from the JAX autodiff FAQ: the diagonal has
    ``r = 0`` and ``d sqrt(x)/dx`` is infinite there. Substituting a safe value
    *before* the sqrt keeps the gradient finite; the caller masks the diagonal
    out of the forward pass.
    """
    n = q.shape[0]
    diff = q[:, None, :] - q[None, :, :]
    r2 = jnp.sum(diff**2, axis=-1)
    self_pair = jnp.eye(n, dtype=bool)
    r = jnp.sqrt(jnp.where(self_pair, 1.0, r2))
    return r, self_pair


def pairwise_sum(phi, q: Array) -> Array:
    """``V(q) = Σ_{i<j} φ(r_ij)`` for any scalar function ``φ`` of distance.

    Only the ``N(N-1)/2`` unique pairs are evaluated (``triu_indices`` is
    static, so this costs nothing at run time). Since no pair has ``r = 0``,
    the gradient is finite without any masking. ``φ`` is mapped over pairs
    with ``jax.vmap``, so it can be a plain function *or* an Equinox MLP that
    takes a scalar.
    """
    i, j = jnp.triu_indices(q.shape[0], k=1)
    r = jnp.sqrt(jnp.sum((q[i] - q[j]) ** 2, axis=-1))
    return jnp.sum(jax.vmap(phi)(r))


def smooth_cutoff(r: Array, r_on: float = 2.0, r_cut: float = 2.5) -> Array:
    """C² switching function: 1 for ``r < r_on``, 0 for ``r > r_cut``.

    Multiplying a pair potential by this makes energy *and* force go smoothly
    to zero, which matters for energy conservation in Verlet.
    """
    x = jnp.clip((r - r_on) / (r_cut - r_on), 0.0, 1.0)
    return 1.0 - x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def lennard_jones(r: Array, epsilon: float = 1.0, sigma: float = 1.0) -> Array:
    """Switched 12-6 Lennard-Jones pair energy ``4ε[(σ/r)¹² − (σ/r)⁶]·S(r)``."""
    sr6 = (sigma / r) ** 6
    return 4.0 * epsilon * (sr6**2 - sr6) * smooth_cutoff(r)


class LennardJones(eqx.Module):
    """Ground-truth potential energy of a Lennard-Jones cluster."""

    epsilon: float = 1.0
    sigma: float = 1.0

    def pair(self, r: Array) -> Array:
        return lennard_jones(r, self.epsilon, self.sigma)

    def __call__(self, q: Array) -> Array:
        return pairwise_sum(self.pair, q)


def forces(potential, q: Array) -> Array:
    """``F = -∇V(q)`` by reverse-mode autodiff, for *any* potential callable."""
    return -jax.grad(potential)(q)


def hexagonal_cluster(n_side: int = 4, spacing: float = 2.0 ** (1 / 6)) -> Array:
    """A roughly round 2-D triangular-lattice patch near the LJ minimum spacing."""
    i, j = jnp.meshgrid(jnp.arange(n_side), jnp.arange(n_side), indexing="ij")
    x = spacing * (i + 0.5 * (j % 2))
    y = spacing * (jnp.sqrt(3.0) / 2.0) * j
    q = jnp.stack([x.ravel(), y.ravel()], axis=-1)
    return q - q.mean(axis=0)


def maxwell_boltzmann(key: PRNGKeyArray, n: int, dim: int, temperature: float) -> Array:
    """Momenta (unit mass) at ``temperature`` with zero net momentum and spin."""
    p = jnp.sqrt(temperature) * jax.random.normal(key, (n, dim))
    return p - p.mean(axis=0)


def remove_rotation(q: Array, p: Array) -> Array:
    """Subtract rigid rotation so 2-D angular momentum is zero."""
    q = q - q.mean(axis=0)
    L = jnp.sum(q[:, 0] * p[:, 1] - q[:, 1] * p[:, 0])
    inertia = jnp.sum(q**2)
    omega = L / inertia
    return p - omega * jnp.stack([-q[:, 1], q[:, 0]], axis=-1)
