import equinox as eqx
import jax
import jax.numpy as jnp
from dynamical_systems.continuous import AbstractODE, solve_ode
from jaxtyping import ArrayLike, Float
from ott.utils import batched_vmap

from .dataset import TimeSeriesDataset


def generate_ode_dataset(
    ode: AbstractODE,
    t_burnin: float,
    t_span: tuple[float, float],
    dt: float,
    n_data: int | None,
    u0: Float[ArrayLike, "n_data dim"] | None = None,
    *,
    seed: int = 0,
    batch_size: int | None = None,
    **solve_ode_kwargs,
) -> TimeSeriesDataset:
    """
    Generates TimeSeriesDataset from a given AbstractODE by numerically integrating from
    initial conditions.

    For chaotic systems, t_burnin can be used to first allow the trajectories to reach
    the attractor before actually generating the dataset trajectories.
    """
    match n_data, u0:
        case n_data, None:
            u0 = jax.random.normal(jax.random.key(seed), shape=(n_data, ode.dim))
        case None, u0:
            ...
        case _:
            raise ValueError(
                """Expected either n_data as int and u0 as None, or n_data as None and 
                u0 as an ArrayLike object"""
            )
    if batch_size is None:
        solve_ode_batch = eqx.filter_jit(
            eqx.filter_vmap(
                lambda ts, u0_: solve_ode(ode, ts, u0_, **solve_ode_kwargs),
                in_axes=(None, 0),
            )
        )
    else:
        solve_ode_batch = eqx.filter_jit(
            batched_vmap(
                lambda ts, u0_: solve_ode(ode, ts, u0_, **solve_ode_kwargs),
                in_axes=(None, 0),
                batch_size=batch_size,
            )
        )
    u0_burnedin = solve_ode_batch(jnp.asarray([0, t_burnin]), u0)[:, -1]
    t_data = jnp.arange(*t_span, dt)
    u_data = solve_ode_batch(t_data, u0_burnedin)
    return TimeSeriesDataset(t_data, u_data)
