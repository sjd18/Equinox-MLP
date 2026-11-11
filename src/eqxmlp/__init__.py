"""eqxmlp: learn a potential energy with an Equinox MLP, then integrate its forces.

V_θ = MLP            (Equinox)
F_θ = -∇V_θ          (jax.grad)
q, p ← Verlet(F_θ)   (jax.lax.scan)
"""

from .data import Dataset, generate_dataset, initial_state
from .diagnostics import angular_momentum, kinetic_energy, radial_distribution, total_energy, total_momentum
from .integrators import Trajectory, rollout, simulate, verlet_step
from .models import DirectForceMLP, GlobalPotentialMLP, PairPotentialMLP
from .systems import (
    LennardJones,
    PhaseSpace,
    forces,
    hexagonal_cluster,
    lennard_jones,
    maxwell_boltzmann,
    pair_distances,
    pairwise_sum,
    remove_rotation,
    smooth_cutoff,
)
from .timelapse import render_timelapse, snapshot_steps
from .training import evaluate, loss_fn, train

__version__ = "0.1.0"

__all__ = [
    "PhaseSpace",
    "LennardJones",
    "lennard_jones",
    "smooth_cutoff",
    "pair_distances",
    "pairwise_sum",
    "forces",
    "hexagonal_cluster",
    "maxwell_boltzmann",
    "remove_rotation",
    "PairPotentialMLP",
    "GlobalPotentialMLP",
    "DirectForceMLP",
    "Trajectory",
    "verlet_step",
    "rollout",
    "simulate",
    "Dataset",
    "initial_state",
    "generate_dataset",
    "loss_fn",
    "train",
    "evaluate",
    "kinetic_energy",
    "total_energy",
    "total_momentum",
    "angular_momentum",
    "radial_distribution",
    "snapshot_steps",
    "render_timelapse",
]
