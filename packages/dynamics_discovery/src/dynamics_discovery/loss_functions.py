import abc
from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from dynamical_systems.analysis.jacobian import jacobian
from dynamical_systems.continuous import AbstractODE
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

    def __init__(
        self,
        ode_true: AbstractODE,
        jacobian_weight: float = 1.0,
        batch_size: int | None = None,
    ):
        self.jacobian_fn = lambda t, u: jacobian(ode_true, t, u)
        self.jacobian_weight = jacobian_weight
        self.batch_size = batch_size

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
        return mse_total + self.jacobian_weight * jac_loss_total, {
            "mse": mse_total,
            "jac_loss": jac_loss_total,
        }
