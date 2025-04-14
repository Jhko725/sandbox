from pathlib import Path
from typing import Any, Generic, Self, TypeVar

import diffrax as dfx
import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from .continuous import AbstractODE, solve_ode
from .utils import get_name, is_arraylike_scalar


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

    def __eq__(self, other: Self) -> bool:
        return eqx.tree_equal(self, other, typematch=True).item()

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

    def save(self, savepath: Path | str, *, overwrite: bool = False) -> None:
        savepath = Path(savepath).with_suffix(".hdf5")
        savepath.parent.mkdir(parents=True, exist_ok=True)

        filemode = "w" if overwrite else "w-"
        with h5py.File(savepath, filemode) as f:
            f.create_dataset("t", data=self.t)
            f.create_dataset("u", data=self.u)
            if self.u0 is not None:
                f.create_dataset("u0", data=self.u0)
            for k, v in self.metadata.items():
                f.attrs[k] = v

    @classmethod
    def load(cls, loadpath: Path | str, *, to_jax: bool = True) -> Self:
        with h5py.File(loadpath, "r") as f:
            t = f["t"][()]
            u = f["u"][()]
            try:
                u0 = f["u0"][()]
            except KeyError:
                u0 = None
            metadata = {
                k: v.item() if is_arraylike_scalar(v) else v for k, v in f.attrs.items()
            }
        dataset = cls(t, u, u0, metadata)
        return dataset.to_jax() if to_jax else dataset
