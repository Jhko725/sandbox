from typing import Any, Generic, TypeVar

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from .continuous import AbstractODE, solve_ode
from .utils import get_name


T = TypeVar("T", bound=ArrayLike)


class TimeSeriesDataset(eqx.Module, Generic[T]):
    """A class to hold time series data from numerical simulations or experiments.

    The class is deliberately designed to be an equinox Module to make it a PyTree:
    this facilitates interactions with jax transforms such as jax.tree.map, etc.
    For an example, see the method TimeSeriesDataset.to_numpy() and
    TimeSeriesDataset.to_jax().
    """

    t: Float[T, " time"]
    u: Float[T, "batch time dim"]
    u0: Float[T, "batch dim"] | None = None
    metadata: dict[str, Any] | None = eqx.field(static=True, default=None)

    def to_numpy(self) -> "TimeSeriesDataset[np.ndarray]":
        return jax.tree.map(np.asarray, self)

    def to_jax(self) -> "TimeSeriesDataset[Array]":
        return jax.tree.map(jnp.asarray, self)

    @classmethod
    def from_dynamical_system(
        cls,
        dynamics: AbstractODE,
        t: Float[T, " time"],
        u0: Float[T, "batch dim"],
        solver: dfx.AbstractAdaptiveSolver = dfx.Tsit5(),
        rtol: float = 1e-7,
        atol: float = 1e-7,
        max_steps: int | None = None,
        **diffeqsolve_kwargs,
    ):
        t = jnp.asarray(t)
        u0 = jnp.asarray(u0)

        @eqx.filter_vmap
        def _solve_ode(u0_: Float[T, " dim"]):
            return solve_ode(
                dynamics,
                t,
                u0_,
                solver,
                rtol,
                atol,
                max_steps=max_steps,
                **diffeqsolve_kwargs,
            )

        u = _solve_ode(u0)

        metadata = {
            "dynamics": get_name(dynamics),
            "solver": get_name(solver),
            "rtol": rtol,
            "atol": atol,
        }
        return cls(t, u, u0, metadata)
