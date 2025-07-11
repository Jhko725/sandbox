from dataclasses import replace
from functools import partial
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax.internal as lxi
import numpy as np
from jax.experimental import jet
from jaxtyping import Array, Float, PyTree
from ott.utils import batched_vmap
from sklearn.neighbors import NearestNeighbors

from ..custom_types import FloatScalar
from ..loss_functions import AbstractDynamicsLoss
from .neuralode import NeuralODE


def create_neighborhood_dataset(
    t: Float[Array, " time"],
    u: Float[Array, "time dim"],
    *,
    num_neighbors: int = 25,
    train_length: int = 2,
    train_length_neighbors: int = 2,
    min_radius: float = 0.05,
    max_radius: float = 0.5,
    adjacent_exclusion_threshold: int = 5,
) -> tuple[
    Float[Array, " time-train_length train_length"],
    Float[Array, "time-train_length train_length dim"],
    Float[Array, "time-train_length train_length_neighbors num_neighbors dim"],
    Float[Array, "time-train_length num_neighbors"],
]:
    """
    Given time series data (t, u), create dataset that can be used to train the
    NeuralNeighborhoodFlow model.

    """

    estimator = NearestNeighbors(radius=max_radius)
    estimator.fit(u[: -train_length_neighbors + 1])

    neigh_dists, neigh_inds = estimator.radius_neighbors(
        u[: -train_length + 1], max_radius, sort_results=True
    )

    neigh_final = []
    lens = []
    for i, (dists, indices) in enumerate(zip(neigh_dists, neigh_inds)):
        criteria = np.logical_and(
            dists >= min_radius, np.abs(indices - i) > adjacent_exclusion_threshold
        )
        selected_inds = indices[criteria]
        lens.append(len(selected_inds))

        if len(selected_inds) >= num_neighbors:
            selected_inds = selected_inds[:num_neighbors]
        else:
            selected_inds = np.pad(
                selected_inds,
                (0, num_neighbors - len(selected_inds)),
                constant_values=i,
            )
        neigh_final.append(selected_inds)

    lens = jnp.clip(jnp.asarray(lens, dtype=jnp.int_), max=num_neighbors)
    idx_nn = jnp.asarray(neigh_final, dtype=jnp.int_)

    def _slice(x, start_index: int):
        return jax.lax.dynamic_slice_in_dim(
            x, start_index, train_length_neighbors, axis=0
        )

    def _inner(i: int, arg=None):
        del arg
        t_slice = jax.lax.dynamic_slice_in_dim(t, i, train_length, axis=0)
        u_slice = jax.lax.dynamic_slice_in_dim(u, i, train_length, axis=0)
        u_nn_slice = jax.vmap(_slice, in_axes=(None, 0), out_axes=1)(u, idx_nn[i])
        weight = jnp.arange(num_neighbors) < lens[i]
        return i + 1, (
            t_slice,
            u_slice,
            u_nn_slice - jnp.expand_dims(u_slice[:train_length_neighbors], 1),
            weight,
        )

    _, dataset = jax.lax.scan(_inner, 0, length=idx_nn.shape[0])
    return dataset


def seminorm(u: tuple[Array, Array]) -> float:
    return lxi.rms_norm(u[0])


class NeuralNeighborhoodFlow(eqx.Module):
    ode: NeuralODE
    stepsize_controller: dfx.AbstractStepSizeController = eqx.field(static=True)
    use_seminorm: bool = eqx.field(static=True)
    second_order: bool = eqx.field(static=True)
    use_taylor_mode: bool = eqx.field(static=True)

    def __init__(
        self,
        ode: NeuralODE,
        use_seminorm: bool = True,
        second_order: bool = False,
        use_taylor_mode: bool = False,
    ):
        self.ode = ode
        self.second_order = second_order
        self.use_seminorm = use_seminorm
        self.use_taylor_mode = use_taylor_mode
        stepsize_controller = self.ode.stepsize_controller
        if self.use_seminorm and isinstance(stepsize_controller, dfx.PIDController):
            self.stepsize_controller = replace(stepsize_controller, norm=seminorm)
        else:
            self.stepsize_controller = stepsize_controller

    @property
    def solver(self) -> dfx.AbstractSolver:
        return self.ode.solver

    @property
    def dt0(self) -> float | None:
        return self.ode.dt0

    def rhs(
        self, t, u: tuple[Float[Array, " dim"], Float[Array, " neighbors dim"]], args
    ):
        y, Dy = u
        z = y + Dy

        def rhs_y(y_):
            return self.ode.rhs(t, y_, args)

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
    batch_size: int | None = None
    weight: float = 1.0

    def __call__(
        self,
        model: NeuralNeighborhoodFlow,
        batch: PyTree[Float[Array, "batch ..."]],
        args: Any = None,
        **kwargs: Any,
    ) -> FloatScalar:
        t_data: Float[Array, "batch time_batch"]
        u_data: Float[Array, "batch time_batch dim"]
        du_data: Float[Array, "batch time_batch_neighbors neighbors dim"]
        weights: Float[Array, "batch neighbors"]
        t_data, u_data, du_data, weights = batch

        batch_size = u_data.shape[0] if self.batch_size is None else self.batch_size
        len_neighbors = du_data.shape[1]

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

        u_pred, du_pred = _solve(t_data[:, :len_neighbors], u_data[:, 0], du_data[:, 0])

        u_nn_pred = jnp.expand_dims(u_pred, 2) + du_pred
        u_nn_data = jnp.expand_dims(u_data[:, :len_neighbors], 2) + du_data
        mse_neighbors = jnp.mean(
            jnp.sum(
                jnp.mean((u_nn_pred - u_nn_data) ** 2, axis=(1, 3)) * weights, axis=-1
            )
            # / jnp.clip(
            #     jnp.sum(weights, axis=-1), min=1
            # )  # Trick do avoid divide by zero
        )

        u_pred_rest = _solve_ode(t_data[:, len_neighbors - 1 :], u_pred[:, -1])
        u_pred_total = jnp.concatenate((u_pred, u_pred_rest[:, 1:]), axis=1)
        mse_total = jnp.mean((u_pred_total - u_data) ** 2)
        return (mse_total + self.weight * mse_neighbors) / (1 + self.weight), {
            "mse": mse_total,
            "mse_neighbors": mse_neighbors,
        }

        # return (mse_total + n_neighbors * mse_neighbors) / (n_neighbors + 1), {
        #     "mse": mse_total,
        #     "mse_neighbors": mse_neighbors,
        # }
