import logging
from copy import replace
from functools import partial
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import scipy.spatial as scspatial
from dynamical_systems.continuous.ode_base import (
    _infer_stepsize_controller,
)
from dynamical_systems.metrics.measure_distances import maximum_mean_discrepancy
from einops import rearrange
from jax.experimental import jet
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

from dynamics_discovery.custom_types import FloatScalar, IntScalarLike
from dynamics_discovery.data import TimeSeriesDataset
from dynamics_discovery.data.loaders import (
    AbstractBatching,
    SegmentLoader,
)

from ..custom_types import FloatScalar
from ..data import TimeSeriesDataset
from ..data.loaders import AbstractBatching, SegmentLoader
from ..loss_functions import AbstractDynamicsLoss
from ..models.neuralode import NeuralODE
from .radius_selection import find_r_max, find_r_min


logger = logging.getLogger(__name__)


class AdjacencyMatrix(eqx.Module):
    indptr: Float[Array, " points+1"]
    indices: Float[Array, " total_neighbors"]
    """Adjacency matrix stored in CSR sparse array format."""


@partial(jax.jit, static_argnames="n_samples")
def sample_neighbor_inds(
    sample_inds: IntScalarLike | Int[Array, " batch"],
    adjacency_matrix: AdjacencyMatrix,
    n_samples: int,
    num_neighbor_cutoff: int = 30,
    key: PRNGKeyArray = jax.random.key(0),
) -> tuple[Int[Array, " n_samples"], bool]:
    sample_inds = jnp.asarray(sample_inds)
    in_shape = sample_inds.shape
    sample_inds: Int[Array, " batch"] = jnp.atleast_1d(sample_inds)

    low, high = jax.vmap(
        lambda x: jax.lax.dynamic_slice_in_dim(adjacency_matrix.indptr, x, 2),
        out_axes=-1,
    )(sample_inds)

    sufficient_neighbors: Bool[Array, " batch"] = high - low > num_neighbor_cutoff

    ptrs: Int[Array, "batch n_samples"] = jax.random.randint(
        key,
        minval=jnp.expand_dims(low, -1),
        maxval=jnp.expand_dims(high, -1),
        shape=(sample_inds.shape[0], n_samples),
    )

    inds: Int[Array, "batch n_samples"] = jax.vmap(
        jax.vmap(
            lambda x: jax.lax.dynamic_index_in_dim(
                adjacency_matrix.indices, x, keepdims=False
            )
        ),
    )(ptrs)

    inds = jnp.where(
        jnp.expand_dims(sufficient_neighbors, -1),
        inds,
        jnp.expand_dims(sample_inds, -1),
    )

    return inds.reshape(in_shape + (n_samples,)), sufficient_neighbors.reshape(in_shape)


class NeighborhoodSegmentLoader(SegmentLoader):
    r_min: float = eqx.field(static=True)
    r_max: float = eqx.field(static=True)
    num_neighbors: int = eqx.field(static=True)
    neighbor_cutoff: int = eqx.field(static=True)

    _adjacency_matrix: AdjacencyMatrix

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        segment_length: int,
        num_neighbors: int,
        r_min_kwargs: dict[str, Any],
        r_max_kwargs: dict[str, Any],
        batch_strategy: AbstractBatching,
        neighbor_cutoff: int | None = None,
        efficient_neighborhood: bool = False,
    ):
        if efficient_neighborhood:
            batch_size_new = batch_strategy.batch_size // num_neighbors
            batch_strategy = replace(batch_strategy, batch_size=batch_size_new)
            num_neighbors = num_neighbors - 1

        super().__init__(dataset, segment_length, batch_strategy)
        logger.info("Calculating minimum radius for neighbor selection")
        self.r_min = find_r_min(dataset, **r_min_kwargs)
        logger.info(f"Minimum radius calculated: r_min={self.r_min}")

        logger.info("Calculating maximum radius for neighbor selection")
        self.r_max = find_r_max(dataset, r_min=self.r_min, **r_max_kwargs)[0]
        logger.info(f"Maximum radius calculated: r_max={self.r_max}")

        self.num_neighbors = num_neighbors
        self.neighbor_cutoff = (
            num_neighbors if neighbor_cutoff is None else neighbor_cutoff
        )

        self._adjacency_matrix = self._build_adjacency_matrix()

    def _build_adjacency_matrix(self):
        tree = scspatial.KDTree(
            rearrange(
                self.dataset.u[:, : -self.segment_length + 1],
                "traj time dim -> (traj time) dim",
            )
        )
        distance_matrix = tree.sparse_distance_matrix(tree, self.r_max)
        within_range = distance_matrix >= self.r_min
        return AdjacencyMatrix(within_range.indptr, within_range.indices)

    def init(self) -> PyTree:
        """
        Returns the initial loader_state to be fed into the first call of
        `self.load_batch`.

        This is inspired by optax's optimizer.init function.
        """
        batch_state_init = self.batch_strategy.init(self.num_total_segments)
        key = jax.random.key(0)
        return (batch_state_init, key)

    def load_batch(
        self, loader_state: PRNGKeyArray
    ) -> tuple[
        Float[Array, "{self.batch_size} {self.segment_length}"],
        Float[Array, "{self.batch_size} {self.segment_length} dim"],
        Float[
            Array, "{self.batch_size} {self.segment_length} {self.num_neighbors} dim"
        ],
    ]:
        (batch_state, key) = loader_state
        linear_indices, batch_state_next = self.batch_strategy.generate_batch(
            batch_state
        )

        sample_indices = self.linear_to_sample_indices(linear_indices)
        t_batch, u_batch = self.get_segments(*sample_indices)

        key, key_next = jax.random.split(key)
        neighbor_linear_indices, mask = sample_neighbor_inds(
            linear_indices,
            self._adjacency_matrix,
            self.num_neighbors,
            self.neighbor_cutoff,
            key,
        )

        _, u_nn_batch = eqx.filter_vmap(self.get_segments)(
            *self.linear_to_sample_indices(neighbor_linear_indices)
        )
        u_nn_batch = rearrange(
            u_nn_batch, "batch neigh time dim -> batch time neigh dim"
        )

        loader_state_next = (batch_state_next, key_next)
        return (t_batch, u_batch, u_nn_batch, mask), loader_state_next


class NeuralNeighborhoodFlow(eqx.Module):
    ode: NeuralODE
    order: int = eqx.field(static=True)
    use_taylor_mode: bool = eqx.field(static=True)

    def __init__(
        self,
        ode: NeuralODE,
        order: int = 2,
        use_taylor_mode: bool = False,
    ):
        self.ode = ode
        self.order = order
        self.use_taylor_mode = use_taylor_mode

    def __check_init__(self):
        if self.order <= 0:
            raise ValueError("Order must be a positive integer")
        elif self.order > 2:
            raise ValueError("Currently, the maximum supported order is 2")

    @property
    def solver(self) -> dfx.AbstractSolver:
        return self.ode.solver

    @property
    def dt0(self) -> float | None:
        return self.ode.dt0

    @property
    def atol(self) -> float | None:
        return self.ode.atol

    @property
    def rtol(self) -> float | None:
        return self.ode.rtol

    def rhs(
        self, t, u: tuple[Float[Array, " dim"], Float[Array, " neighbors dim"]], args
    ):
        y, Dy = u
        z = y + Dy

        def rhs_y(y_):
            return self.ode.rhs(t, y_, args)

        """TODO: Abstract this part by writing a function that returns the truncated 
        Taylor expansion of a given function"""

        def _first_order(y_, z_):
            return eqx.filter_jvp(rhs_y, (y_,), (z_ - y_,))

        if self.use_taylor_mode:

            def _second_order(y_, z_):
                Dy_ = z_ - y_
                dy, dDys = jet.jet(rhs_y, (y_,), ((Dy_, jnp.zeros_like(Dy_)),))
                dDy = dDys[0] + 0.5 * dDys[1]
                return dy, dDy
        else:

            def _second_order(y_, z_):
                (dy, dDy_1), (_, dDy_2) = eqx.filter_jvp(
                    lambda y__: _first_order(y__, z_), (y_,), (z_ - y_,)
                )
                dDy = 1.5 * dDy_1 + 0.5 * dDy_2
                return dy, dDy

        if self.order == 1:
            dy, dDy = eqx.filter_vmap(
                _first_order, in_axes=(None, 0), out_axes=(None, 0)
            )(y, z)
        else:
            dy, dDy = eqx.filter_vmap(
                _second_order, in_axes=(None, 0), out_axes=(None, 0)
            )(y, z)
        return dy, dDy

    def solve(
        self,
        ts: Float[Array, " time"],
        u0: tuple[Float[Array, " dim"], Float[Array, " neighbors dim"]],
        args=None,
        **kwargs,
    ) -> tuple[Float[Array, " time dim"], Float[Array, " time neighbors dim"]]:
        stepsize_controller = _infer_stepsize_controller(
            self.dt0, self.rtol, self.atol, ts
        )
        sol = dfx.diffeqsolve(
            dfx.ODETerm(self.rhs),
            self.solver,
            ts[0],
            ts[-1],
            self.dt0,
            u0,
            args,
            saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=stepsize_controller,
            **kwargs,
        )
        return sol.ys


class NeighborhoodMSELoss(AbstractDynamicsLoss):
    neighbor_traj_length: int = eqx.field(static=True)
    weight: float = eqx.field(static=True, default=1.0)
    taylor_order: int | None = eqx.field(static=True, default=2)
    use_taylor_mode: bool = eqx.field(static=True, default=False)
    multiterm: bool = eqx.field(static=True, default=False)

    def propagate_neighborhood(
        self,
        model: NeuralODE,
        t: Float[Array, " time_neigh"],
        u0: Float[Array, " dim"],
        u_neigh0: Float[Array, " neighbors dim"],
        args,
        **kwargs,
    ) -> tuple[
        Float[Array, "time_neigh dim"], Float[Array, "time_neigh neighbors dim"]
    ]:
        if self.taylor_order is None:
            # Use vmapped model to propagate the neighbor positions;
            # amounts to just doing batched solve over the neighborhood
            u0_total = jnp.concatenate((u0.reshape(1, -1), u_neigh0), axis=0)

            def _solve(u_):
                return model.solve(t, u_, args, max_steps=16384, **kwargs)

            u_total = eqx.filter_vmap(_solve, out_axes=1)(u0_total)
            return u_total[:, 0], u_total[:, 1:]

        else:
            model_nn = NeuralNeighborhoodFlow(
                model, self.taylor_order, self.use_taylor_mode
            )
            du0 = u_neigh0 - u0
            u_pred, du_pred = model_nn.solve(
                t, (u0, du0), args, max_steps=16384, **kwargs
            )
            u_nn_pred = rearrange(u_pred, "T D -> T 1 D") + du_pred
            return u_pred, u_nn_pred

    def __call__(
        self,
        model: NeuralODE,
        batch: PyTree[Float[Array, "batch ..."]],
        args: Any = None,
        **kwargs: Any,
    ) -> tuple[FloatScalar | list[FloatScalar], dict[str, Array]]:
        t_data: Float[Array, "batch time"]
        u_data: Float[Array, "batch time dim"]
        u_nn_data: Float[Array, "batch time_neighbors neighbors dim"]
        mask: Bool[Array, "batch"]

        t_data, u_data, u_nn_data, mask = batch
        u_nn_data = u_nn_data[:, : self.neighbor_traj_length]

        segment_length = u_data.shape[1]

        # Solve for neighborhood positions along time
        @eqx.filter_vmap
        def _propagate_neighborhood(t_, u0_, u_neigh0_):
            return self.propagate_neighborhood(
                model, t_, u0_, u_neigh0_, args, **kwargs
            )

        u_pred, u_nn_pred = _propagate_neighborhood(
            t_data[:, : self.neighbor_traj_length], u_data[:, 0], u_nn_data[:, 0]
        )

        # If solving for center trajectories for longer time, additionally solve and
        # concatenate
        if segment_length > self.neighbor_traj_length:
            u_pred_rest = eqx.filter_vmap(partial(model.solve, args=args, **kwargs))(
                t_data[:, self.neighbor_traj_length - 1 :], u_pred[:, -1]
            )
            u_pred = jnp.concatenate((u_pred, u_pred_rest[:, 1:]), axis=1)

        # MSE loss for the trajectory
        loss_traj = jnp.mean((u_pred - u_data) ** 2)

        # Compute neighborhood loss using either the neighbor positions or the
        # displacement from the center point to the neighbors
        def neighborhood_loss(
            neigh_pred: Float[Array, "batch time_neigh neighbors dim"],
            neigh_data: Float[Array, "batch time_neigh neighbors dim"],
        ) -> Float[Array, ""]:
            loss_per_sample: Float[Array, " batch"] = jnp.mean(
                jax.vmap(jax.vmap(maximum_mean_discrepancy))(neigh_pred, neigh_data),
                axis=1,
            )
            return jnp.sum(loss_per_sample * mask) / jnp.clip(jnp.sum(mask), min=1)

        loss_neigh = neighborhood_loss(u_nn_pred, u_nn_data)

        if self.multiterm:
            loss = [loss_traj, self.weight * loss_neigh]
        else:
            loss = loss_traj + self.weight * loss_neigh
        return loss, {"mse": loss_traj, "loss_neighbors": loss_neigh}


class EfficientNeighborhoodMSELoss(NeighborhoodMSELoss):
    neighbor_traj_length: int = eqx.field(static=True)
    weight: float = eqx.field(static=True)
    taylor_order: int | None = eqx.field(static=True)
    use_taylor_mode: bool = eqx.field(static=True)
    multiterm: bool = eqx.field(static=True)

    def __init__(
        self,
        neighbor_traj_length: int,
        weight: float = 1.0,
        taylor_order: int | None = 2,
        use_taylor_mode: bool = False,
        multiterm: bool = False,
    ):
        super().__init__(
            neighbor_traj_length=neighbor_traj_length,
            weight=weight,
            taylor_order=taylor_order,
            use_taylor_mode=use_taylor_mode,
            multiterm=multiterm,
        )

    def __call__(
        self,
        model: NeuralODE,
        batch: PyTree[Float[Array, "batch ..."]],
        args: Any = None,
        **kwargs: Any,
    ) -> tuple[FloatScalar | list[FloatScalar], dict[str, Array]]:
        t_data: Float[Array, "batch time"]
        u_data: Float[Array, "batch time dim"]
        u_nn_data: Float[Array, "batch time_neighbors neighbors dim"]
        mask: Bool[Array, "batch"]

        t_data, u_data, u_nn_data, mask = batch
        u_nn_data = u_nn_data[:, : self.neighbor_traj_length]

        segment_length = u_data.shape[1]

        # Solve for neighborhood positions along time

        @eqx.filter_vmap
        def _propagate_neighborhood(t_, u0_, u_neigh0_):
            return self.propagate_neighborhood(
                model, t_, u0_, u_neigh0_, args, **kwargs
            )

        u_pred, u_nn_pred = _propagate_neighborhood(
            t_data[:, : self.neighbor_traj_length], u_data[:, 0], u_nn_data[:, 0]
        )
        u_nn_data_ = rearrange(u_nn_data, "b t n d -> (b n) t d")
        u_pred_rest = eqx.filter_vmap(
            partial(model.solve, args=args, **kwargs),
            in_axes=(None, 0),
        )(t_data[0, : self.neighbor_traj_length], u_nn_data_[:, 0])
        # If solving for center trajectories for longer time, additionally solve and
        # concatenate
        # if segment_length > self.neighbor_traj_length:
        #     u_pred_rest = batched_vmap(
        #         partial(model.solve, args=args, **kwargs), batch_size=batch_size
        #     )(t_data[:, self.neighbor_traj_length - 1 :], u_pred[:, -1])
        #     u_pred = jnp.concatenate((u_pred, u_pred_rest[:, 1:]), axis=1)

        # MSE loss for the trajectory
        loss_traj = jnp.mean(
            jnp.concatenate(
                (
                    jnp.mean((u_pred - u_data) ** 2, axis=(1, 2)),
                    jnp.mean((u_pred_rest - u_nn_data_) ** 2, axis=(1, 2)),
                )
            )
        )

        # Compute neighborhood loss using either the neighbor positions or the
        # displacement from the center point to the neighbors
        def neighborhood_loss(
            neigh_pred: Float[Array, "batch time_neigh neighbors dim"],
            neigh_data: Float[Array, "batch time_neigh neighbors dim"],
        ) -> Float[Array, ""]:
            loss_per_sample: Float[Array, " batch"] = jnp.mean(
                jax.vmap(jax.vmap(maximum_mean_discrepancy))(neigh_pred, neigh_data),
                axis=1,
            )
            return jnp.sum(loss_per_sample * mask) / jnp.clip(jnp.sum(mask), min=1)

        loss_neigh = neighborhood_loss(u_nn_pred, u_nn_data)

        if self.multiterm:
            loss = [loss_traj, self.weight * loss_neigh]
        else:
            loss = loss_traj + self.weight * loss_neigh
        return loss, {"mse": loss_traj, "loss_neighbors": loss_neigh}
