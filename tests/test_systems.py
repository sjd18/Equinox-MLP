import jax
import jax.numpy as jnp
import numpy as np
import pytest

import eqxmlp as em


def brute_lj_forces(q, eps=1.0, sigma=1.0):
    """Analytic forces for the switched LJ potential, written out by hand."""
    n = q.shape[0]
    F = np.zeros_like(q)
    dphi = jax.grad(lambda r: em.lennard_jones(r, eps, sigma))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = q[i] - q[j]
                r = np.linalg.norm(d)
                F[i] -= float(dphi(r)) * d / r
    return F


def test_pairwise_sum_matches_brute_force():
    q = em.hexagonal_cluster(4) + 0.05 * jax.random.normal(jax.random.PRNGKey(0), (16, 2))
    r = np.linalg.norm(q[:, None] - q[None], axis=-1)
    iu = np.triu_indices(16, 1)
    np.testing.assert_allclose(em.LennardJones()(q), np.sum(em.lennard_jones(r[iu])), rtol=1e-12)


def test_autodiff_forces_match_hand_written():
    q = em.hexagonal_cluster(3) + 0.05 * jax.random.normal(jax.random.PRNGKey(1), (9, 2))
    np.testing.assert_allclose(em.forces(em.LennardJones(), q), brute_lj_forces(np.asarray(q)), atol=1e-10)


@pytest.mark.parametrize("r0", [2.0, 2.5])
def test_cutoff_is_smooth(r0):
    """Value, first and second derivative are continuous at both switch points."""
    for f in [em.smooth_cutoff, jax.grad(em.smooth_cutoff), jax.grad(jax.grad(em.smooth_cutoff))]:
        assert abs(f(r0 - 1e-9) - f(r0 + 1e-9)) < 1e-6


def test_potential_vanishes_beyond_cutoff():
    assert em.lennard_jones(jnp.array(2.6)) == 0.0
    assert jax.grad(em.lennard_jones)(jnp.array(2.6)) == 0.0


def test_initial_state_has_no_drift_or_spin():
    s = em.initial_state(jax.random.PRNGKey(0))
    np.testing.assert_allclose(em.total_momentum(s.p), 0.0, atol=1e-12)
    np.testing.assert_allclose(em.angular_momentum(s.q, s.p), 0.0, atol=1e-12)


def test_radial_distribution_is_normalised():
    s = em.initial_state(jax.random.PRNGKey(0))
    r, dens = em.radial_distribution(s.q[None], r_max=10.0, n_bins=200)
    np.testing.assert_allclose(jnp.sum(dens) * (r[1] - r[0]), 1.0, rtol=1e-12)
