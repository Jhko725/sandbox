from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Self, TypeVar

import diffrax as dfx
import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from dynamical_systems.continuous import (
    AbstractODE,
)
from dynamical_systems.utils import (
    get_name,
    is_arraylike_scalar,
)
from jaxtyping import Array, ArrayLike, Float


T = TypeVar("T", bound=ArrayLike)


class TimeSeriesDataset(eqx.Module, Sequence[T]):
    """A class to hold time series data from numerical simulations or experiments.

    The class is deliberately designed to be an equinox Module to make it a PyTree:
    this facilitates interactions with jax transforms such as jax.tree.map, etc.
    For an example, see the method TimeSeriesDataset.to_numpy() and
    TimeSeriesDataset.to_jax().
    """

    t: Float[T, "?batch time"]
    u: Float[T, "batch time dim"]
    u0: Float[T, "batch dim"] | None = None
    metadata: dict[str, Any] | None = eqx.field(static=True)

    def __init__(
        self,
        t: Float[T, "?batch time"] | Float[T, " time"],
        u: Float[T, "batch time dim"],
        u0: Float[T, "batch dim"] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        # In principle, should use np.at_least2d if t is a numpy array
        self.t = jnp.atleast_2d(t)
        self.u = u
        self.u0 = u0
        self.metadata = metadata

    def __eq__(self, other: Self) -> bool:
        return eqx.tree_equal(self, other, typematch=True).item()

    def __len__(self) -> int:
        """Returns the number of time points in the dataset."""
        return self.u.shape[1]

    def __getitem__(self, idx) -> Self:
        """Returns a sub-dataset by indexing along the time axis.

        All index types supported by numpy and jax should work."""
        t, u = self.t[:, idx], self.u[:, idx]

        if self.u0 is None:
            u0 = None
        else:
            if u.ndim < 2:
                u0 = u
            else:
                u0 = u[:, 0]

        return replace(self, t=t, u=u, u0=u0)

    def to_numpy(self) -> "TimeSeriesDataset[np.ndarray]":
        return jax.tree.map(np.asarray, self)

    def to_jax(self) -> "TimeSeriesDataset[Array]":
        return jax.tree.map(jnp.asarray, self)

    @classmethod
    def from_dynamical_system(
        cls,
        dynamics: AbstractODE,
        t: Float[T, "?batch time"],
        u0: Float[T, "batch dim"],
        t_burnin: float = 0.0,
        solver: dfx.AbstractAdaptiveSolver = dfx.Tsit5(),
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int | None = None,
        **diffeqsolve_kwargs,
    ):
        t = jnp.asarray(t)
        u0 = jnp.asarray(u0)

        @eqx.filter_vmap
        def _solve_ode(u0_: Float[T, " dim"]):
            return dynamics.solve(
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
