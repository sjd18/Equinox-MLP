"""Training data: frames from ground-truth molecular dynamics.

Several independent trajectories are run in parallel with ``jax.vmap`` over
random initial velocities, sampled every ``save_every`` steps, and labelled
with the true energy and true forces (``-jax.grad`` of the true potential).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from .integrators import simulate
from .systems import LennardJones, PhaseSpace, forces, hexagonal_cluster, maxwell_boltzmann, remove_rotation

__all__ = ["Dataset", "initial_state", "generate_dataset"]


class Dataset(NamedTuple):
    q: Array  # (B, N, dim) positions
    energy: Array  # (B,) potential energy
    forces: Array  # (B, N, dim)

    def __len__(self) -> int:
        return self.q.shape[0]


def initial_state(key: PRNGKeyArray, n_side: int = 4, temperature: float = 0.35) -> PhaseSpace:
    """Hexagonal patch of ``n_side²`` atoms with thermal, spin-free velocities."""
    k_pos, k_vel = jax.random.split(key)
    q = hexagonal_cluster(n_side)
    q = q + 0.02 * jax.random.normal(k_pos, q.shape)
    p = maxwell_boltzmann(k_vel, q.shape[0], q.shape[1], temperature)
    return PhaseSpace(q, remove_rotation(q, p))


def generate_dataset(
    key: PRNGKeyArray,
    n_traj: int = 32,
    n_steps: int = 4000,
    save_every: int = 40,
    dt: float = 0.005,
    temperature: float = 0.35,
    burn_in: int = 10,
    potential=LennardJones(),
) -> Dataset:
    """Sample frames from ``n_traj`` ground-truth trajectories.

    Returns ``n_traj * (n_steps // save_every + 1 - burn_in)`` labelled frames.
    """
    keys = jax.random.split(key, n_traj)
    # Each trajectory starts at a different temperature so the data covers a
    # range of energies (and so of interatomic distances).
    temps = temperature * jnp.linspace(0.6, 1.4, n_traj)
    states = jax.vmap(initial_state, in_axes=(0, None, 0))(keys, 4, temps)
    force_fn = lambda q: forces(potential, q)  # noqa: E731
    run = jax.vmap(lambda s: simulate(force_fn, s, dt=dt, n_steps=n_steps, save_every=save_every).q)
    q = run(states)[:, burn_in:]
    q = q.reshape(-1, *q.shape[2:])
    return Dataset(q=q, energy=jax.vmap(potential)(q), forces=jax.vmap(force_fn)(q))
