from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Self

import equinox as eqx
import h5py
import numpy as np
from dynamical_systems.transforms import Standardize
from jaxtyping import Float


class TimeSeriesDataset(eqx.Module):
    """
    Class representing the collection of trajectories from a dynamical system.

    All data manipulation is done with numpy instead of jax.numpy to not overload the
    GPU RAM.
    """

    t: Float[np.ndarray, "samples time"]
    u: Float[np.ndarray, "samples time dim"]

    def __init__(
        self,
        t: Float[np.ndarray, "#samples time"],
        u: Float[np.ndarray, "samples time dim"],
    ):
        t = np.atleast_2d(np.asarray(t))
        self.t = np.tile(t, (u.shape[0], 1)) if t.shape[0] == 1 else t
        self.u = np.asarray(u)

    def __check_init__(self):
        if self.t.shape != self.u.shape[:2]:
            raise ValueError("t and u do not have compatible shapes!")

    @property
    def dt(self) -> float:
        # Assumes that all trajectories are equispaced with the same time increment
        return self.t[0, 1] - self.t[0, 0]

    @property
    def trajectory_length(self) -> int:
        return self.t.shape[1]

    def __len__(self) -> int:
        return self.t.shape[0]

    def __getitem__(self, idx):
        return np.take(self.t, idx, axis=0), np.take(self.u, idx, axis=0)

    def downsample(self, downsample_factor: int) -> Self:
        return replace(
            self, t=self.t[:, ::downsample_factor], u=self.u[:, ::downsample_factor]
        )

    def add_noise(self, noise_std_relative: float, *, seed: int = 0) -> Self:
        rng = np.random.default_rng(seed)
        std = np.std(self.u, axis=(0, 1))
        noise = rng.normal(size=self.u.shape) * std * noise_std_relative
        return replace(self, u=self.u + noise)

    def standardize(self) -> tuple[Self, Standardize]:
        """Standarization is performed using the statistics of the full dataset."""
        mean = np.mean(self.u, axis=(0, 1))
        std = np.std(self.u, axis=(0, 1))
        transform = Standardize(mean, std)
        dataset_new = self.apply_transform(transform)
        return dataset_new, transform

    def destandardize(self, transform: Standardize) -> Self:
        return self.apply_transform(transform.inverse)

    def apply_transform(
        self,
        transform: Callable[
            [Float[np.ndarray, "samples time dim"]],
            Float[np.ndarray, "samples time dim"],
        ],
    ):
        u_transformed = transform(self.u)
        return replace(self, u=u_transformed)

    @classmethod
    def from_hdf5(cls, filepath: str | Path) -> Self:
        with h5py.File(filepath, "r") as f:
            t = f["t"][()]
            u = f["u"][()]

        return cls(t, u)

    def to_hdf5(self, savepath: str | Path) -> None:
        savepath = Path(savepath).with_suffix(".hdf5")
        savepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(savepath, "w") as f:
            f.create_dataset("t", data=self.t)
            f.create_dataset("u", data=self.u)
