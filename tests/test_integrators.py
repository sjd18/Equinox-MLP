import jax
import jax.numpy as jnp
import numpy as np

import eqxmlp as em

LJ = em.LennardJones()
F = lambda q: em.forces(LJ, q)  # noqa: E731


def test_verlet_conserves_energy_and_momenta():
    s = em.initial_state(jax.random.PRNGKey(0))
    tr = em.simulate(F, s, dt=0.005, n_steps=10_000, save_every=100)
    E = em.total_energy(LJ, tr)
    assert jnp.abs(E - E[0]).max() / jnp.abs(E[0]) < 1e-3
    np.testing.assert_allclose(em.total_momentum(tr.p), 0.0, atol=1e-12)
    np.testing.assert_allclose(em.angular_momentum(tr.q, tr.p), 0.0, atol=1e-12)


def test_time_reversible():
    s = em.initial_state(jax.random.PRNGKey(1))
    fwd = em.simulate(F, s, dt=0.005, n_steps=200)
    back = em.simulate(F, em.PhaseSpace(fwd.q[-1], fwd.p[-1]), dt=-0.005, n_steps=200)
    np.testing.assert_allclose(back.q[-1], s.q, atol=1e-9)


def test_save_every_consistent():
    s = em.initial_state(jax.random.PRNGKey(2))
    a = em.simulate(F, s, dt=0.005, n_steps=100)
    b = em.simulate(F, s, dt=0.005, n_steps=100, save_every=20)
    assert b.q.shape[0] == 6
    np.testing.assert_allclose(a.q[::20], b.q, atol=1e-13)


def test_force_is_cached_one_eval_per_step():
    """Velocity Verlet should call the force function once per step (plus once at init)."""
    calls = []

    def counting(q):
        calls.append(1)
        return F(q)

    s = em.initial_state(jax.random.PRNGKey(3))
    carry = em.integrators._Carry(s.q, s.p, counting(s.q))
    jax.make_jaxpr(lambda c: em.verlet_step(counting, c, 0.005))(carry)
    assert len(calls) == 2  # one at init, one traced inside the step
