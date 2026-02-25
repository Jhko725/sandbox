from functools import partial
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import scipy.spatial as scspatial
from dynamical_systems.continuous import AbstractODE
from dynamical_systems.continuous.ode_base import (
    _infer_stepsize_controller,
    AbstractODE,
)
from dynamical_systems.metrics.measure_distances import maximum_mean_discrepancy
from einops import rearrange
from jax.experimental import jet
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PRNGKeyArray, PyTree
from ott.utils import batched_vmap

from dynamics_discovery.custom_types import FloatScalar, IntScalarLike
from dynamics_discovery.data import TimeSeriesDataset
from dynamics_discovery.data.loaders import (
    AbstractBatching,
    SegmentLoader,
)

from .custom_types import FloatScalar
from .data import TimeSeriesDataset
from .data.loaders import AbstractBatching, SegmentLoader
from .loss_functions import AbstractDynamicsLoss
from .models.abstract import AbstractDynamicsModel
from .models.neuralode import NeuralODE


class AdjacencyMatrix(eqx.Module):
    indptr: Float[ArrayLike, " points+1"]
    indices: Float[ArrayLike, " total_neighbors"]
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
        r_min: float,
        r_max: float,
        num_neighbors: int,
        neighbor_cutoff: int,
        batch_strategy: AbstractBatching,
    ):
        super().__init__(dataset, segment_length, batch_strategy)
        self.r_min = r_min
        self.r_max = r_max
        self.num_neighbors = num_neighbors
        self.neighbor_cutoff = neighbor_cutoff

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
    weight: float = 1.0
    batch_size: int | None = None
    multiterm: bool = False
    taylor_order: int | None = eqx.field(static=True, default=2)
    use_taylor_mode: bool = False
    neighbor_position_loss: bool = eqx.field(static=True, default=False)

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
                return model.solve(t, u_, args, **kwargs)

            u_total = eqx.filter_vmap(_solve, out_axes=1)(u0_total)
            return u_total[:, 0], u_total[:, 1:]

        else:
            model_nn = NeuralNeighborhoodFlow(
                model, self.taylor_order, self.use_taylor_mode
            )
            du0 = u_neigh0 - u0
            u_pred, du_pred = model_nn.solve(t, (u0, du0), args, **kwargs)
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

        batch_size = u_data.shape[0] if self.batch_size is None else self.batch_size
        segment_length = u_data.shape[1]

        # Solve for neighborhood positions along time
        @partial(batched_vmap, batch_size=batch_size)
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
            u_pred_rest = batched_vmap(
                partial(model.solve, args=args, **kwargs), batch_size=batch_size
            )(t_data[:, self.neighbor_traj_length - 1 :], u_pred[:, -1])
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

        if self.neighbor_position_loss:
            loss_neigh = neighborhood_loss(u_nn_pred, u_nn_data)
        else:
            du_nn_pred = u_nn_pred - rearrange(
                u_pred[:, : self.neighbor_traj_length], "B T D -> B T 1 D"
            )
            du_nn_data = u_nn_data - rearrange(
                u_data[:, : self.neighbor_traj_length], "B T D -> B T 1 D"
            )
            loss_neigh = neighborhood_loss(du_nn_pred, du_nn_data)

        if self.multiterm:
            loss = [loss_traj, self.weight * loss_neigh]
        else:
            loss = loss_traj + self.weight * loss_neigh
        return loss, {"mse": loss_traj, "loss_neighbors": loss_neigh}


class NormalODE(AbstractODE):
    ode: AbstractODE

    @property
    def dim(self) -> int:
        return self.ode.dim * (self.ode.dim + 1)

    @property
    def solver(self) -> dfx.AbstractSolver:
        return self.ode.solver

    @property
    def dt0(self) -> float | None:
        return self.ode.dt0

    @property
    def stepsize_controller(self) -> dfx.AbstractStepSizeController:
        return self.ode.stepsize_controller

    def rhs(self, t, u: tuple[Float[Array, " dim"], Float[Array, " dim"]], args):
        x, Nx = u

        def rhs_ode(x_):
            return self.ode.rhs(t, x_, args)

        dx, vjp_fun = jax.vjp(rhs_ode, x)
        dNx = -vjp_fun(Nx)[0]

        return dx, dNx

    def solve(
        self,
        ts: Float[Array, " time"],
        u0: tuple[Float[Array, " dim"], Float[Array, "dim"]],
        args=None,
        **kwargs,
    ) -> tuple[Float[Array, " time dim"], Float[Array, " time dim"]]:
        sol = dfx.diffeqsolve(
            dfx.ODETerm(self.rhs),
            self.solver,
            ts[0],
            ts[-1],
            self.dt0,
            u0,
            args,
            saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=self.stepsize_controller,
            **kwargs,
        )
        return sol.ys


def get_normal_direction(points: Float[Array, "batch dim"]):
    points_centered = points  # - jnp.mean(points, axis=0)
    _, _, Q_T = jnp.linalg.svd(points_centered, full_matrices=False)
    return Q_T[2]


get_normal_direction_batch = jax.vmap(jax.vmap(get_normal_direction))


class NormalVectorSegmentLoader(NeighborhoodSegmentLoader):
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        segment_length: int,
        num_neighbors: int,
        batch_strategy: AbstractBatching,
    ):
        super().__init__(dataset, segment_length, num_neighbors, batch_strategy)

    def load_batch(
        self, loader_state: PRNGKeyArray
    ) -> tuple[
        Float[Array, "{self.batch_size} {self.segment_length}"],
        Float[Array, "{self.batch_size} {self.segment_length} dim"],
        Float[
            Array, "{self.batch_size} {self.segment_length} {self.num_neighbors} dim"
        ],
    ]:
        (t_batch, u_batch, u_nn_batch), loader_state_next = super().load_batch(
            loader_state
        )
        normal_batch = get_normal_direction_batch(u_nn_batch)
        return (t_batch, u_batch, normal_batch), loader_state_next


def directional_cosine_squared(x1, x2):
    denom = jnp.sum(x1 * x1) * jnp.sum(x2 * x2)
    return jnp.dot(x1, x2) ** 2 / denom


class NormalLoss(AbstractDynamicsLoss):
    weight: float = 1.0
    batch_size: int | None = None
    multiterm: bool = False
    orthogonal_loss: bool = False

    def __call__(
        self,
        model: AbstractDynamicsModel,
        batch: PyTree[Float[Array, "batch ..."]],
        args: Any = None,
        **kwargs: Any,
    ) -> FloatScalar:
        t_data, u_data, Nu_data = batch
        batch_size = u_data.shape[0] if self.batch_size is None else self.batch_size

        normal_ode = NormalODE(model)

        @partial(batched_vmap, in_axes=(0, 0, 0), batch_size=batch_size)
        def _loss(t_data_: Float[Array, " time"], u_data_, Nu_data_):
            u_pred, Nu_pred = normal_ode.solve(
                t_data_, (u_data_[0], Nu_data_[0]), args, **kwargs
            )
            mse_u = jnp.mean((u_pred - u_data_) ** 2)
            sin_sqr = 1 - jax.vmap(directional_cosine_squared)(Nu_data_, Nu_pred)

            if self.orthogonal_loss:
                ortho = jax.vmap(directional_cosine_squared)(
                    jax.vmap(lambda t_, u_: model.ode.rhs(t_, u_, None))(
                        t_data_, u_data_
                    ),
                    Nu_data_,
                )
                ortho_loss = jnp.mean(ortho)
            else:
                ortho_loss = None
            return mse_u, jnp.mean(sin_sqr), ortho_loss

        mse_, sin_sqr_, ortho_ = _loss(t_data, u_data, Nu_data)
        mse_total = jnp.mean(mse_)
        sin_sqr_total = jnp.mean(sin_sqr_)

        if self.orthogonal_loss:
            ortho_total = jnp.mean(ortho_)
            if self.multiterm:
                loss = [
                    mse_total,
                    self.weight * sin_sqr_total,
                    self.weight * ortho_total,
                ]
            else:
                loss = (
                    mse_total + self.weight * sin_sqr_total + self.weight * ortho_total
                )

            aux_dict = {
                "mse": mse_total,
                "normal_loss": sin_sqr_total,
                "ortho_loss": ortho_total,
            }

        else:
            if self.multiterm:
                loss = [mse_total, self.weight * sin_sqr_total]
            else:
                loss = mse_total + self.weight * sin_sqr_total

            aux_dict = {
                "mse": mse_total,
                "normal_loss": sin_sqr_total,
            }

        return loss, aux_dict
