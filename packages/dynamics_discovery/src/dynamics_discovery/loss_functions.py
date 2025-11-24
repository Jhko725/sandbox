import abc
from collections.abc import Callable
from functools import partial
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from dynamical_systems.analysis.jacobian import jacobian
from dynamical_systems.continuous import AbstractODE, TangentODE
from jaxtyping import Array, Float, PyTree
from ott.utils import batched_vmap

from .custom_types import FloatScalar
from .models.abstract import AbstractDynamicsModel


class AbstractDynamicsLoss(eqx.Module, strict=True):
    """
    A protocol defining the function signature for loss functions that are compatible
    with the training methods in this library.
    """

    @abc.abstractmethod
    def __call__(
        self,
        model: AbstractDynamicsModel,
        batch: PyTree[Float[Array, "batch ..."]],
        args: Any,
        **kwargs: Any,
    ) -> FloatScalar: ...


class MSELoss(AbstractDynamicsLoss):
    batch_size: int | None = None

    def __call__(
        self,
        model: AbstractDynamicsModel,
        batch: PyTree[Float[Array, "batch ..."]],
        args: Any = None,
        **kwargs: Any,
    ) -> FloatScalar:
        t_data, u_data = batch

        batch_size = u_data.shape[0] if self.batch_size is None else self.batch_size

        @partial(batched_vmap, in_axes=(0, 0), batch_size=batch_size)
        def _mse(t_data_: Float[Array, " time"], u_data_):
            u_pred = model.solve(t_data_, u_data_[0], args, **kwargs)
            return jnp.mean((u_pred - u_data_) ** 2)

        mse_batch = _mse(t_data, u_data)
        mse_total = jnp.mean(mse_batch)
        return mse_total, {"mse": mse_total}


class JacobianMatchingMSE(AbstractDynamicsLoss):
    jacobian_fn: Callable = eqx.field(static=True)
    jacobian_weight: float
    batch_size: int | None
    multiterm: bool

    def __init__(
        self,
        ode_true: AbstractODE,
        jacobian_weight: float = 1.0,
        batch_size: int | None = None,
        multiterm: bool = False,
    ):
        self.jacobian_fn = lambda t, u: jacobian(ode_true, t, u)
        self.jacobian_weight = jacobian_weight
        self.batch_size = batch_size
        self.multiterm = multiterm

    def __call__(
        self,
        model: AbstractDynamicsModel,
        batch: PyTree[Float[Array, "batch ..."]],
        args: Any = None,
        **kwargs: Any,
    ) -> FloatScalar:
        t_data, u_data = batch

        batch_size = u_data.shape[0] if self.batch_size is None else self.batch_size

        @partial(batched_vmap, in_axes=(0, 0), batch_size=batch_size)
        def _mse(t_data_: Float[Array, " time"], u_data_):
            u_pred = model.solve(t_data_, u_data_[0], args, **kwargs)
            return jnp.mean((u_pred - u_data_) ** 2)

        @partial(batched_vmap, in_axes=(0, 0), batch_size=batch_size)
        def _jac_loss(t_data_: Float[Array, " time"], u_data_):
            jac_pred = jacobian(model, t_data_[0], u_data_[0])
            jac_true = self.jacobian_fn(t_data_[0], u_data_[0])
            return jnp.mean((jac_pred - jac_true) ** 2)

        mse_total = jnp.mean(_mse(t_data, u_data))
        jac_loss_total = jnp.mean(_jac_loss(t_data, u_data))

        if self.multiterm:
            loss = [mse_total, self.jacobian_weight * jac_loss_total]
        else:
            loss = mse_total + self.jacobian_weight * jac_loss_total

        return loss, {
            "mse": mse_total,
            "jac_loss": jac_loss_total,
        }


def tangent_evolution_matrix(
    ode: AbstractODE,
    x: Float[Array, " dim"],
    ts: Float[Array, " time"],
    solver: dfx.AbstractAdaptiveSolver = dfx.Tsit5(),
    stepsize_controller: dfx.AbstractAdaptiveStepSizeController = dfx.PIDController(
        rtol=1e-4, atol=1e-6
    ),
):
    if isinstance(ode, AbstractODE):
        tangent_ode = TangentODE(ode)
        u0 = (x, jnp.identity(ode.dim))

        sol = dfx.diffeqsolve(
            dfx.ODETerm(tangent_ode.rhs),
            solver,
            ts[0],
            ts[-1],
            None,
            u0,
            None,
            saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=stepsize_controller,
        )
        u, M_t = sol.ys
        return u, M_t[1:]  # remove time dimension
    else:
        u0 = (x, jnp.identity(ode.dim))

        @partial(jax.vmap, in_axes=(None, -1), out_axes=(None, -1))
        def rhs_jac(x_, Tx_i):
            return jax.jvp(lambda u: ode.step(0.0, 1.0, u), (x_,), (Tx_i,))

        u1 = rhs_jac(*u0)

        return jax.tree.map(lambda a, b: jnp.stack((a, b), axis=0), u0, u1)


class TangentEvolutionMatchingMSE(AbstractDynamicsLoss):
    ode_true: AbstractODE = eqx.field(static=True)
    weight: float
    batch_size: int | None
    multiterm: bool

    def __init__(
        self,
        ode_true: AbstractODE,
        weight: float = 1.0,
        batch_size: int | None = None,
        multiterm: bool = False,
    ):
        self.ode_true = ode_true
        self.weight = weight
        self.batch_size = batch_size
        self.multiterm = multiterm

    def __call__(
        self,
        model: AbstractDynamicsModel,
        batch: PyTree[Float[Array, "batch ..."]],
        args: Any = None,
        **kwargs: Any,
    ) -> FloatScalar:
        t_data, u_data = batch

        batch_size = u_data.shape[0] if self.batch_size is None else self.batch_size
        print(self)

        @partial(batched_vmap, in_axes=(0, 0), batch_size=batch_size)
        def _loss(t_data_: Float[Array, " time"], u_data_):
            u_pred, evol_pred = tangent_evolution_matrix(
                model,
                u_data_[0],
                t_data_,
                # model.solver,
                # model.stepsize_controller,
            )
            _, evol_true = tangent_evolution_matrix(
                self.ode_true,
                u_data_[0],
                t_data_,
                # model.solver,
                # model.stepsize_controller,
            )
            mse = jnp.mean((u_pred - u_data_) ** 2)
            return mse, jnp.mean((evol_pred - evol_true) ** 2)

        mse_, evol_loss_ = _loss(t_data, u_data)
        mse_total = jnp.mean(mse_)
        evol_loss_total = jnp.mean(evol_loss_)

        if self.multiterm:
            loss = [mse_total, evol_loss_total]
        else:
            loss = mse_total + self.weight * evol_loss_total
        return loss, {
            "mse": mse_total,
            "tangent_evolution_loss": evol_loss_total,
        }


class PushforwardMatchingMSE(AbstractDynamicsLoss):
    weight: float
    batch_size: int | None = None
    multiterm: bool = False

    def __call__(
        self,
        model: AbstractDynamicsModel,
        batch: PyTree[Float[Array, "batch ..."]],
        args: Any = None,
        **kwargs: Any,
    ) -> FloatScalar:
        (t_data, u_data), (M1, M2) = batch

        batch_size = u_data.shape[0] if self.batch_size is None else self.batch_size

        @partial(batched_vmap, in_axes=(0, 0, 0, 0), batch_size=batch_size)
        def _loss(t_data_: Float[Array, " time"], u_data_, M1_, M2_):
            u_pred, evol_pred = tangent_evolution_matrix(
                model,
                u_data_[0],
                t_data_,
                # model.solver,
                # model.stepsize_controller,
            )
            mse = jnp.mean((u_pred - u_data_) ** 2)

            return mse, jnp.mean((M1_.T @ evol_pred[0].T - M2_) ** 2)

        mse_, DF_loss_ = _loss(t_data, u_data, M1, M2)
        mse_total = jnp.mean(mse_)
        DF_loss_total = jnp.mean(DF_loss_)

        if self.multiterm:
            loss = [mse_total, DF_loss_total]
        else:
            loss = mse_total + self.weight * DF_loss_total
        return loss, {
            "mse": mse_total,
            "DF_loss": DF_loss_total,
        }
