"""Velocity Verlet driven by an arbitrary force field, rolled out with lax.scan.

The integrator only needs a ``force_fn(q) -> F``. That can be
``-grad`` of the true potential, ``-grad`` of a learned Equinox MLP, or the
raw output of a network that predicts forces directly; the integrator does
not care, which makes it a fair test bed for all three.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .systems import PhaseSpace

__all__ = ["Trajectory", "verlet_step", "rollout", "simulate"]


class Trajectory(NamedTuple):
    t: Array
    q: Array  # (T, N, dim)
    p: Array  # (T, N, dim)

    @property
    def states(self) -> PhaseSpace:
        return PhaseSpace(self.q, self.p)


class _Carry(NamedTuple):
    q: Array
    p: Array
    f: Array  # force at q, reused by the next step's first half-kick


def verlet_step(force_fn, carry: _Carry, dt, mass=1.0) -> _Carry:
    """Kick–drift–kick. One force evaluation per step."""
    p_half = carry.p + 0.5 * dt * carry.f
    q_new = carry.q + dt * p_half / mass
    f_new = force_fn(q_new)
    p_new = p_half + 0.5 * dt * f_new
    return _Carry(q_new, p_new, f_new)


def rollout(step_fn, init, n_steps: int, save_every: int = 1):
    """Apply ``step_fn`` ``n_steps`` times via nested ``lax.scan``.

    Returns ``[init, c_s, c_2s, ...]`` stacked along a new leading axis, with
    ``s = save_every``. The inner scan emits nothing, so memory is
    ``O(n_steps / save_every)``.
    """
    if n_steps % save_every:
        raise ValueError(f"n_steps={n_steps} must be divisible by save_every={save_every}")

    def inner(c, _):
        return step_fn(c), None

    def outer(c, _):
        c, _ = jax.lax.scan(inner, c, length=save_every)
        return c, c

    _, saved = jax.lax.scan(outer, init, length=n_steps // save_every)
    return jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), init, saved)


@eqx.filter_jit
def simulate(force_fn, state0: PhaseSpace, *, dt, n_steps: int, save_every: int = 1, mass=1.0) -> Trajectory:
    """Integrate ``q̈ = F(q)/m`` with velocity Verlet.

    ``force_fn`` may be a closure or an Equinox module; ``eqx.filter_jit``
    traces its arrays (e.g. MLP weights) and treats everything else as static.
    """
    init = _Carry(state0.q, state0.p, force_fn(state0.q))
    out = rollout(lambda c: verlet_step(force_fn, c, dt, mass), init, n_steps, save_every)
    t = dt * save_every * jnp.arange(n_steps // save_every + 1)
    return Trajectory(t, out.q, out.p)
