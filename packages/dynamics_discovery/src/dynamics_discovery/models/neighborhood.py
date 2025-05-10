from dataclasses import replace
from functools import partial
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax.internal as lxi
import scipy.spatial as scspatial
from jaxtyping import Array, Float, PyTree

from ..custom_types import FloatScalar
from ..loss_functions import AbstractDynamicsLoss
from .neuralode import NeuralODE


def create_neighborhood_dataset(
    t: Float[Array, " time"],
    u: Float[Array, "time dim"],
    *,
    num_neighbors: int = 25,
    train_length: int = 2,
) -> tuple[
    Float[Array, " time-train_length train_length"],
    Float[Array, "time-train_length train_length dim"],
    Float[Array, "time-train_length num_neighbors train_length dim"],
]:
    """
    Given time series data (t, u), create dataset that can be used to train the
    NeuralNeighborhoodFlow model.

    """
    u_query = u[:-train_length]
    kdtree = scspatial.KDTree(u_query)
    idx_nn = jnp.asarray(
        kdtree.query(u_query, k=num_neighbors + 1)[1][:, 1:], dtype=jnp.int_
    )

    def _slice(x, start_index: int):
        return jax.lax.dynamic_slice_in_dim(x, start_index, train_length, axis=0)

    def _inner(i: int, arg=None):
        del arg
        t_slice = _slice(t, i)
        u_slice = _slice(u, i)
        u_nn_slice = jax.vmap(_slice, in_axes=(None, 0))(u, idx_nn[i])
        return i + 1, (t_slice, u_slice, u_nn_slice - u_slice)

    _, dataset = jax.lax.scan(_inner, 0, length=len(t) - train_length)
    return dataset


def seminorm(u: tuple[Array, Array]) -> float:
    return lxi.rms_norm(u[0])


class NeuralNeighborhoodFlow(eqx.Module):
    ode: NeuralODE
    stepsize_controller: dfx.AbstractStepSizeController = eqx.field(static=True)
    use_seminorm: bool = eqx.field(static=True)
    second_order: bool = eqx.field(static=False)

    def __init__(
        self, ode: NeuralODE, use_seminorm: bool = True, second_order: bool = False
    ):
        self.ode = ode
        self.second_order = second_order
        self.use_seminorm = use_seminorm
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

    def rhs(self, t, u, args):
        y, Dy = u
        z = y + Dy

        def rhs_y(y_):
            return self.ode.rhs(t, y_, args)

        def first_order(y_):
            return eqx.filter_jvp(rhs_y, (y_,), (z - y_,))

        if not self.second_order:
            dy, dDy = first_order(y)
        else:
            (dy, dDy_1), (_, dDy_2) = eqx.filter_jvp(first_order, (y,), (Dy,))
            dDy = 1.5 * dDy_1 + 0.5 * dDy_2
        return dy, dDy

    def solve(self, ts: Float[Array, " time"], u0, args=None, **kwargs):
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
    def __call__(
        self,
        model: NeuralNeighborhoodFlow,
        batch: PyTree[Float[Array, "batch ..."]],
        args: Any = None,
        **kwargs: Any,
    ) -> FloatScalar:
        t_data: Float[Array, "batch time_batch"]
        u_data: Float[Array, "batch time_batch dim"]
        du_data: Float[Array, "batch neighbors time_batch dim"]
        t_data, u_data, du_data = batch

        @partial(eqx.filter_vmap, in_axes=(0, 0, 0))
        @partial(eqx.filter_vmap, in_axes=(None, None, 0), out_axes=(None, 0))
        def _solve(
            t: Float[Array, " time_batch"],
            u0: Float[Array, " dim"],
            du0: Float[Array, " dim"],
        ):
            return model.solve(
                t,
                (u0, du0),
                **kwargs,
            )

        n_neighbors = du_data.shape[1]
        u_pred, du_pred = _solve(t_data, u_data[:, 0], du_data[:, :, 0])
        mse_total = jnp.mean((u_pred - u_data) ** 2)
        u_nn_pred = jnp.expand_dims(u_pred, 1) + du_pred
        u_nn_data = jnp.expand_dims(u_data, 1) + du_data
        mse_neighbors = jnp.mean((u_nn_pred - u_nn_data) ** 2)
        return (mse_total + n_neighbors * mse_neighbors) / (n_neighbors + 1), {
            "mse": mse_total,
            "mse_neighbors": mse_neighbors,
        }
