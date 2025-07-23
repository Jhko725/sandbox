from functools import partial
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import scipy.spatial as scspatial
from jax.experimental import jet
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from ott.utils import batched_vmap

from .custom_types import FloatScalar
from .data import TimeSeriesDataset
from .data.loaders import AbstractSegmentLoader
from .loss_functions import AbstractDynamicsLoss
from .models.neuralode import NeuralODE


class NeighborhoodSegmentLoader(AbstractSegmentLoader):
    dataset: TimeSeriesDataset
    segment_length: int = eqx.field(static=True)
    num_neighbors: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    seed: int = eqx.field(default=0, static=True)
    _tree: scspatial.KDTree = eqx.field(static=True, init=False)

    def __post_init__(self):
        self._tree = scspatial.KDTree(
            self.dataset.u[:, : -self.segment_length + 1].reshape(
                -1, self.dataset.u.shape[-1]
            )
        )

    def init(self):
        return jax.random.PRNGKey(self.seed)

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
        key, new_loader_state = jax.random.split(loader_state)
        linear_indices = jax.random.randint(
            key, (self.batch_size,), 0, self.num_total_segments
        )
        t_batch, u_batch = self.get_segments(
            *self.linear_to_sample_indices(linear_indices)
        )
        nn_linear_indices: Int[Array, "{self.batch_size} {self.num_neighbors} dim"] = (
            self.get_neighbors_linear_indices(u_batch[:, 0])
        )
        _, u_nn_batch = eqx.filter_vmap(self.get_segments)(
            *self.linear_to_sample_indices(nn_linear_indices)
        )
        u_nn_batch = jnp.permute_dims(u_nn_batch, (0, 2, 1, 3))
        return (t_batch, u_batch, u_nn_batch), new_loader_state


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

    def __call__(
        self,
        model: NeuralNeighborhoodFlow,
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

        @partial(batched_vmap, in_axes=(0, 0, 0), batch_size=batch_size)
        def _solve(
            t: Float[Array, " time_batch_neighbors"],
            u0: Float[Array, " dim"],
            du0: Float[Array, " neighbors dim"],
        ):
            return model.solve(
                t,
                (u0, du0),
                **kwargs,
            )

        @partial(batched_vmap, in_axes=(0, 0), batch_size=batch_size)
        def _solve_ode(
            t: Float[Array, " {time_batch-time_batch_neighbors}"],
            u0: Float[Array, " dim"],
        ):
            return model.ode.solve(
                t,
                u0,
                **kwargs,
            )

        u_pred, du_pred = _solve(
            t_data[:, :len_neighbors], u_data[:, 0], u_nn_data[:, 0]
        )

        u_nn_pred = jnp.expand_dims(u_pred, 2) + du_pred

        # mse_neighbors = jnp.mean(
        #     jnp.sum(
        #         jnp.mean((u_nn_pred - u_nn_data) ** 2, axis=(1, 3)) * weights, axis=-1
        #     )
        #     # / jnp.clip(
        #     #     jnp.sum(weights, axis=-1), min=1
        #     # )  # Trick do avoid divide by zero
        # )
        mse_neighbors = jnp.mean((u_nn_pred - u_nn_data) ** 2)

        # u_pred_rest = _solve_ode(t_data[:, len_neighbors - 1 :], u_pred[:, -1])
        # u_pred_total = jnp.concatenate((u_pred, u_pred_rest[:, 1:]), axis=1)
        # mse_total = jnp.mean((u_pred_total - u_data) ** 2)
        mse_total = jnp.mean((u_pred - u_data) ** 2)
        return (mse_total + self.weight * mse_neighbors) / (1 + self.weight), {
            "mse": mse_total,
            "mse_neighbors": mse_neighbors,
        }

        # return (mse_total + n_neighbors * mse_neighbors) / (n_neighbors + 1), {
        #     "mse": mse_total,
        #     "mse_neighbors": mse_neighbors,
        # }
