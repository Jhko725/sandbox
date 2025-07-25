import abc
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int, PRNGKeyArray, PyTree

from dynamics_discovery.data.dataset import TimeSeriesDataset


def get_trajectory_segments(dataset, sample_idx, time_idx, segment_length: int):
    t_segment = jax.lax.dynamic_index_in_dim(dataset.t, sample_idx, keepdims=False)
    t_segment = jax.lax.dynamic_slice_in_dim(t_segment, time_idx, segment_length)

    u_segment = jax.lax.dynamic_index_in_dim(dataset.u, sample_idx, keepdims=False)
    u_segment = jax.lax.dynamic_slice_in_dim(u_segment, time_idx, segment_length)
    return t_segment, u_segment


class AbstractSegmentLoader(eqx.Module):
    """
    Abstract base class for SegmentLoaders, which are dedicated clases that samples a
    given dataset of trajectories (of class TimeSeriesDataset) and returns a batch of
    trajectory segments with fixed length.

    This class inspired by DataLoader classes in many deep learning libraries, such as
    `torch.utils.data.DataLoader`.
    """

    dataset: eqx.AbstractVar[TimeSeriesDataset]
    segment_length: eqx.AbstractVar[int]
    batch_size: eqx.AbstractVar[int]

    def __check_init__(self):
        if self.segment_length < 2:
            raise ValueError("Minimum allowed segment length is 2.")

    @property
    def num_segments_per_traj(self) -> int:
        return self.dataset.trajectory_length - self.segment_length + 1

    @property
    def num_total_segments(self) -> int:
        return len(self.dataset) * self.num_segments_per_traj

    @property
    def num_batches(self) -> int:
        """Number of batches required to cover (on average) the entire dataset.

        This number is always rounded up to the nearest integer."""
        return math.ceil(self.num_total_segments / self.batch_size)

    def get_segments(self, traj_idx, time_idx):
        return eqx.filter_vmap(get_trajectory_segments, in_axes=(None, 0, 0, None))(
            self.dataset, traj_idx, time_idx, self.segment_length
        )

    def linear_to_sample_indices(
        self, linear_indices: Int[Array, " {self.batch_size}"]
    ) -> tuple[Int[Array, " {self.batch_size}"], Int[Array, " {self.batch_size}"]]:
        """
        Converts the 1D array of linear indices representing the starting position of
        the segments in the batch to a tuple of indices that can be used to locate the
        said position in `self.dataset.u`.
        """
        return jnp.divmod(linear_indices, self.num_segments_per_traj)

    @abc.abstractmethod
    def init(self) -> PyTree:
        """
        Returns the initial loader_state to be fed into the first call of
        `self.load_batch`.

        This is inspired by optax's optimizer.init function.
        """
        ...

    @abc.abstractmethod
    def load_batch(self, loader_state: PyTree) -> tuple[PyTree[Array], PyTree]:
        """
        Main logic to load a single batch of time series data segments.

        loader_state contains any extra state necessary to generate a particular batch:
        For random sampling, this corresponds to the random key, and for minibatch
        sampling, this would correspond to the batch index (and the random key if the
        data is reshuffled each epoch.)
        """
        ...


class AllSegmentLoader(AbstractSegmentLoader):
    dataset: TimeSeriesDataset
    segment_length: int = eqx.field(static=True)

    @property
    def batch_size(self) -> int:
        return self.num_total_segments

    def init(self):
        return None

    def load_batch(self, loader_state=None):
        sample_indices = self.linear_to_sample_indices(
            np.arange(self.num_total_segments, dtype=np.int_)
        )
        return self.get_segments(*sample_indices), loader_state


class RandomSegmentLoader(AbstractSegmentLoader):
    dataset: TimeSeriesDataset
    segment_length: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    seed: int = eqx.field(default=0, static=True)

    def init(self):
        return jax.random.PRNGKey(self.seed)

    def load_batch(self, loader_state: PRNGKeyArray):
        key, new_loader_state = jax.random.split(loader_state)
        linear_indices = jax.random.randint(
            key, (self.batch_size,), 0, self.num_total_segments
        )
        return self.get_segments(
            *self.linear_to_sample_indices(linear_indices)
        ), new_loader_state
