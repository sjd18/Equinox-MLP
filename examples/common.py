"""Shared settings and checkpoint helpers for the example scripts."""

import json
from pathlib import Path

import equinox as eqx
import jax

import eqxmlp as em

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
CKPT = ROOT / "checkpoints"

# Hyperparameters of the main model (also stored next to the checkpoint).
PAIR_HPARAMS = {"width": 48, "depth": 2}
TRAIN = {"steps": 3000, "batch_size": 64, "lr": 3e-3}
DT = 0.005


def datasets():
    """Training frames (32 trajectories) and held-out test frames (8 new ones)."""
    return em.generate_dataset(jax.random.PRNGKey(0)), em.generate_dataset(jax.random.PRNGKey(100), n_traj=8)


def save_pair_model(model, name="pair_potential"):
    CKPT.mkdir(exist_ok=True)
    eqx.tree_serialise_leaves(CKPT / f"{name}.eqx", model)
    (CKPT / f"{name}.json").write_text(json.dumps(PAIR_HPARAMS, indent=2) + "\n")


def load_pair_model(name="pair_potential"):
    """Load the trained pair model, or return None if it hasn't been trained yet."""
    path = CKPT / f"{name}.eqx"
    if not path.exists():
        return None
    hp = json.loads((CKPT / f"{name}.json").read_text())
    skeleton = em.PairPotentialMLP(**hp, key=jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(path, skeleton)
