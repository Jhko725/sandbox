from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from dynamical_systems.analysis import lyapunov_gr
from dynamical_systems.continuous import solve_ode
from dynamical_systems.metrics import (
    cosine_similarity as cosine_similarity,
    maximum_mean_discrepancy as maximum_mean_discrepancy,
    mean_squared_error as mean_squared_error,
    sinkhorn_divergence as sinkhorn_divergence,
)
from ott.utils import batched_vmap

from dynamics_discovery.io import load_model
from dynamics_discovery.models import NeuralODE


def load_experiment(
    exp_type: str,
    key: int,
    downsample: int,
    train_length: int,
    noise: float,
    rootdir,
) -> NeuralODE:
    loaddir = Path(rootdir) / f"downsample={downsample}/len={train_length}/{exp_type}"
    if exp_type == "neighborhood":
        loaddir = loaddir / "weight=1.0"
    elif exp_type == "proxy_tangent_evolution":
        loaddir = loaddir / "weight=0.001"

    model_paths = list(loaddir.glob(f"*_noise={noise}_key={key}*"))
    if len(model_paths) == 0:
        raise ValueError("No models found with the given conditions")
    elif len(model_paths) > 1:
        raise ValueError("Multiple models found with the given conditions!")
    else:
        return load_model(model_paths[0])


@eqx.filter_jit
def calculate_lyapunov(model, t, u0, batch_size: int):
    lyas = batched_vmap(
        lambda u0_: lyapunov_gr(model, u0_, t, rtol=1e-7, atol=1e-7, max_steps=None)[0],
        batch_size=batch_size,
    )(u0)
    return jnp.mean(lyas[:, -1], axis=0)


@eqx.filter_jit
def solve_batch(model, t, u0_batch):
    def _solve(u0_):
        return solve_ode(model, t, u0_, rtol=1e-4, atol=1e-6, max_steps=None)

    return batched_vmap(_solve, batch_size=5000)(u0_batch)
