import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import eqxmlp as em


@pytest.fixture
def q():
    return em.hexagonal_cluster(3) + 0.05 * jax.random.normal(jax.random.PRNGKey(0), (9, 2))


@pytest.fixture
def pair_model():
    return em.PairPotentialMLP(width=16, depth=2, key=jax.random.PRNGKey(1))


def rotation(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s], [s, c]])


def test_pair_model_invariances(pair_model, q):
    """Energy is invariant to translation, rotation and relabelling atoms."""
    E = pair_model.energy(q)
    perm = jax.random.permutation(jax.random.PRNGKey(2), q.shape[0])
    np.testing.assert_allclose(pair_model.energy(q + jnp.array([3.0, -1.0])), E, rtol=1e-12)
    np.testing.assert_allclose(pair_model.energy(q @ rotation(0.7).T), E, rtol=1e-12)
    np.testing.assert_allclose(pair_model.energy(q[perm]), E, rtol=1e-12)


def test_pair_model_forces_are_equivariant(pair_model, q):
    R = rotation(1.1)
    np.testing.assert_allclose(pair_model.forces(q @ R.T), pair_model.forces(q) @ R.T, atol=1e-12)


def test_pair_model_newton_third_law(pair_model, q):
    """Net force and net torque vanish identically, even untrained."""
    Fq = pair_model.forces(q)
    np.testing.assert_allclose(Fq.sum(0), 0.0, atol=1e-12)
    np.testing.assert_allclose(jnp.sum(q[:, 0] * Fq[:, 1] - q[:, 1] * Fq[:, 0]), 0.0, atol=1e-12)


def test_global_model_breaks_translation_symmetry(q):
    m = em.GlobalPotentialMLP(9, key=jax.random.PRNGKey(3))
    assert abs(m.energy(q + 1.0) - m.energy(q)) > 1e-6
    assert jnp.abs(m.forces(q).sum(0)).max() > 1e-6


def test_learned_potential_conserves_its_own_energy_and_momenta(pair_model):
    s = em.initial_state(jax.random.PRNGKey(4), n_side=3, temperature=0.05)
    tr = em.simulate(pair_model.forces, s, dt=0.002, n_steps=2000, save_every=100)
    E = em.total_energy(pair_model.energy, tr)
    assert jnp.abs(E - E[0]).max() < 1e-4
    np.testing.assert_allclose(em.total_momentum(tr.p), 0.0, atol=1e-12)


def test_force_matching_gradient_is_finite_and_second_order(pair_model):
    """The loss contains jax.grad; its parameter gradient needs grad-of-grad."""
    data = em.generate_dataset(jax.random.PRNGKey(5), n_traj=2, n_steps=400, save_every=40, burn_in=0)
    g = eqx.filter_grad(em.loss_fn)(pair_model, data)
    leaves = jax.tree.leaves(eqx.filter(g, eqx.is_array))
    assert all(jnp.all(jnp.isfinite(x)) for x in leaves)
    assert any(jnp.abs(x).max() > 0 for x in leaves)


def test_training_learns_lennard_jones():
    data = em.generate_dataset(jax.random.PRNGKey(6), n_traj=8, n_steps=2000)
    model = em.PairPotentialMLP(width=32, depth=2, key=jax.random.PRNGKey(1))
    model, losses = em.train(model, data, key=jax.random.PRNGKey(8), steps=600, batch_size=32, verbose=False)
    assert losses[-50:].mean() < 0.01 * losses[:10].mean()
    assert em.evaluate(model, data)["force_r2"] > 0.99
    r = jnp.linspace(1.1, 2.0, 10)
    np.testing.assert_allclose(jax.vmap(model.pair)(r), em.LennardJones().pair(r), atol=0.05)


def test_direct_force_model_shape():
    m = em.DirectForceMLP(9, key=jax.random.PRNGKey(9))
    q = em.hexagonal_cluster(3)
    assert m.forces(q).shape == q.shape
    assert not hasattr(m, "energy")


def test_train_callback_sees_every_step():
    data = em.generate_dataset(jax.random.PRNGKey(10), n_traj=2, n_steps=400, save_every=40, burn_in=0)
    seen = []
    model = em.PairPotentialMLP(width=8, depth=1, key=jax.random.PRNGKey(11))
    trained, _ = em.train(
        model,
        data,
        key=jax.random.PRNGKey(12),
        steps=5,
        batch_size=8,
        verbose=False,
        callback=lambda n, m: seen.append((n, m)),
    )
    assert [n for n, _ in seen] == list(range(6))
    assert seen[0][1] is model and seen[-1][1] is trained


def test_render_timelapse_writes_gif(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    data = em.generate_dataset(jax.random.PRNGKey(13), n_traj=2, n_steps=400, save_every=40, burn_in=0)
    keep = em.snapshot_steps(4, n=3)
    snaps = []
    model = em.PairPotentialMLP(width=8, depth=1, key=jax.random.PRNGKey(14))
    _, losses = em.train(
        model,
        data,
        key=jax.random.PRNGKey(15),
        steps=4,
        batch_size=8,
        verbose=False,
        callback=lambda n, m: snaps.append((n, m)) if n in keep else None,
    )
    start = em.initial_state(jax.random.PRNGKey(16), n_side=3)
    out = em.render_timelapse(
        snaps, losses, test_data=data, start=start, path=tmp_path / "t.gif", sim_steps=20, dpi=30
    )
    assert out.exists() and out.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
