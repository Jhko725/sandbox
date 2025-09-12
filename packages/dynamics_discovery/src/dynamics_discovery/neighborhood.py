from functools import partial
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import scipy.spatial as scspatial
from dynamical_systems.continuous import AbstractODE
from jax.experimental import jet
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from ott.utils import batched_vmap

from .custom_types import FloatScalar
from .data import TimeSeriesDataset
from .data.loaders import AbstractBatching, SegmentLoader
from .loss_functions import AbstractDynamicsLoss
from .models.abstract import AbstractDynamicsModel
from .models.neuralode import NeuralODE


class NeighborhoodSegmentLoader(SegmentLoader):
    num_neighbors: int = eqx.field(static=True)
    _tree: scspatial.KDTree = eqx.field(static=True)

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        segment_length: int,
        num_neighbors: int,
        batch_strategy: AbstractBatching,
    ):
        super().__init__(dataset, segment_length, batch_strategy)
        self.num_neighbors = num_neighbors
        self._tree = self._create_tree()

    def _create_tree(self):
        return scspatial.KDTree(
            self.dataset.u[:, : -self.segment_length + 1].reshape(
                -1, self.dataset.u.shape[-1]
            )
        )

    def get_neighbors_linear_indices(
        self, u0_batch: Float[Array, "{self.batch_size} dim"]
    ):
        result_shape = jax.ShapeDtypeStruct(
            (u0_batch.shape[0], self.num_neighbors), jnp.int64
        )
        return jax.pure_callback(
            lambda u0_: self._tree.query(u0_, self.num_neighbors + 1)[1][:, 1:],
            result_shape,
            u0_batch,
            vmap_method="expand_dims",
        )

    def load_batch(
        self, loader_state: PRNGKeyArray
    ) -> tuple[
        Float[Array, "{self.batch_size} {self.segment_length}"],
        Float[Array, "{self.batch_size} {self.segment_length} dim"],
        Float[
            Array, "{self.batch_size} {self.segment_length} {self.num_neighbors} dim"
        ],
    ]:
        (t_batch, u_batch), loader_state_next = super().load_batch(loader_state)
        nn_linear_indices: Int[Array, "{self.batch_size} {self.num_neighbors} dim"] = (
            self.get_neighbors_linear_indices(u_batch[:, 0])
        )
        _, u_nn_batch = eqx.filter_vmap(self.get_segments)(
            *self.linear_to_sample_indices(nn_linear_indices)
        )
        u_nn_batch = jnp.permute_dims(u_nn_batch, (0, 2, 1, 3))
        return (t_batch, u_batch, u_nn_batch), loader_state_next


class NeuralNeighborhoodFlow(eqx.Module):
    ode: NeuralODE
    second_order: bool = eqx.field(static=True)
    use_taylor_mode: bool = eqx.field(static=True)

    def __init__(
        self,
        ode: NeuralODE,
        second_order: bool = False,
        use_taylor_mode: bool = False,
    ):
        self.ode = ode
        self.second_order = second_order
        self.use_taylor_mode = use_taylor_mode

    @property
    def solver(self) -> dfx.AbstractSolver:
        return self.ode.solver

    @property
    def dt0(self) -> float | None:
        return self.ode.dt0

    @property
    def stepsize_controller(self) -> dfx.AbstractStepSizeController:
        return self.ode.stepsize_controller

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

        if not self.second_order:
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


class NeighborhoodMSELoss(AbstractDynamicsLoss):
    weight: float = 1.0
    batch_size: int | None = None
    multiterm: bool = False
    second_order: bool = False
    use_taylor_mode: bool = False

    def __call__(
        self,
        model: NeuralODE,
        batch: PyTree[Float[Array, "batch ..."]],
        args: Any = None,
        **kwargs: Any,
    ) -> FloatScalar:
        t_data: Float[Array, "batch time_batch"]
        u_data: Float[Array, "batch time_batch dim"]
        u_nn_data: Float[Array, "batch time_batch neighbors dim"]
        # weights: Float[Array, "batch neighbors"]
        # t_data, u_data, du_data, weights = batch
        t_data, u_data, u_nn_data = batch

        batch_size = u_data.shape[0] if self.batch_size is None else self.batch_size
        len_neighbors = u_nn_data.shape[1]
        model_nn = NeuralNeighborhoodFlow(
            model, self.second_order, self.use_taylor_mode
        )

        @partial(batched_vmap, in_axes=(0, 0, 0), batch_size=batch_size)
        def _solve(
            t: Float[Array, " time_batch_neighbors"],
            u0: Float[Array, " dim"],
            du0: Float[Array, " neighbors dim"],
        ):
            return model_nn.solve(
                t,
                (u0, du0),
                **kwargs,
            )

        @partial(batched_vmap, in_axes=(0, 0), batch_size=batch_size)
        def _solve_ode(
            t: Float[Array, " {time_batch-time_batch_neighbors}"],
            u0: Float[Array, " dim"],
        ):
            return model.solve(
                t,
                u0,
                **kwargs,
            )

        du_nn_data = u_nn_data - jnp.expand_dims(u_data, 2)
        u_pred, du_pred = _solve(
            t_data[:, :len_neighbors], u_data[:, 0], du_nn_data[:, 0]
        )

        # u_nn_pred = jnp.expand_dims(u_pred, 2) + du_pred

        # mse_neighbors = jnp.mean(
        #     jnp.sum(
        #         jnp.mean((u_nn_pred - u_nn_data) ** 2, axis=(1, 3)) * weights, axis=-1
        #     )
        #     # / jnp.clip(
        #     #     jnp.sum(weights, axis=-1), min=1
        #     # )  # Trick do avoid divide by zero
        # )
        # mse_neighbors = jnp.mean((u_nn_pred - u_nn_data) ** 2)
        mse_neighbors = jnp.mean((du_pred - du_nn_data) ** 2)

        # u_pred_rest = _solve_ode(t_data[:, len_neighbors - 1 :], u_pred[:, -1])
        # u_pred_total = jnp.concatenate((u_pred, u_pred_rest[:, 1:]), axis=1)
        # mse_total = jnp.mean((u_pred_total - u_data) ** 2)
        mse_total = jnp.mean((u_pred - u_data) ** 2)

        if self.multiterm:
            loss = [mse_total, self.weight * mse_neighbors]
        else:
            loss = mse_total + self.weight * mse_neighbors
        return loss, {
            "mse": mse_total,
            "mse_neighbors": mse_neighbors,
        }

        # return (mse_total + n_neighbors * mse_neighbors) / (n_neighbors + 1), {
        #     "mse": mse_total,
        #     "mse_neighbors": mse_neighbors,
        # }


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
