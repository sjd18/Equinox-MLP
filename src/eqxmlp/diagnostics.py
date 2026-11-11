"""Physical diagnostics for trajectories."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .integrators import Trajectory
from .systems import pair_distances

__all__ = [
    "kinetic_energy",
    "total_energy",
    "total_momentum",
    "angular_momentum",
    "radial_distribution",
]


def kinetic_energy(p: Array, mass: float = 1.0) -> Array:
    return 0.5 * jnp.sum(p**2, axis=(-2, -1)) / mass


def total_energy(potential, traj: Trajectory, mass: float = 1.0) -> Array:
    """``T + V`` at each saved frame, for any potential callable ``V(q)``."""
    return kinetic_energy(traj.p, mass) + jax.vmap(potential)(traj.q)


def total_momentum(p: Array) -> Array:
    return jnp.sum(p, axis=-2)


def angular_momentum(q: Array, p: Array) -> Array:
    """2-D angular momentum ``Σ x p_y − y p_x``."""
    return jnp.sum(q[..., 0] * p[..., 1] - q[..., 1] * p[..., 0], axis=-1)


def radial_distribution(q: Array, r_max: float = 3.0, n_bins: int = 120) -> tuple[Array, Array]:
    """Histogram of pair distances over frames ``(T, N, dim)``, normalised per pair.

    For a finite cluster (no box) this is the pair-distance density rather
    than a bulk ``g(r)``, but it is the right fingerprint for comparing two
    simulations of the same system.
    """
    r, self_pair = jax.vmap(pair_distances)(q)
    upper = jnp.triu(~self_pair[0], k=1)
    d = r[:, upper].ravel()
    edges = jnp.linspace(0.0, r_max, n_bins + 1)
    counts, _ = jnp.histogram(d, bins=edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    density = counts / (d.size * (edges[1] - edges[0]))
    return centers, density
