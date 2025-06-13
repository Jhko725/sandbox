import abc
from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from ott.utils import batched_vmap
from dynamical_systems.continuous import AbstractODE
from dynamical_systems.analysis.jacobian import jacobian

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
    analytical_jacobian: Callable = eqx.field(static=True)
    jacobian_weight: float
    batch_size: int | None

    def __init__(
        self,
        ode_true: AbstractODE,
        jacobian_weight: float = 1.0,
        batch_size: int | None = None,
    ):
        self.analytical_jacobian = lambda t, u: jacobian(ode_true, t, u)
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
            u_std = jnp.array([7.82955024, 8.89867481, 8.54572838])
            u_mean = jnp.array([1.3145725, 1.33083936, 23.60976671])

            jac_pred = jacobian(model, t_data_[0], u_data_[0])
            jac_true = (
                jnp.diag(1 / u_std)
                @ self.analytical_jacobian(t_data_[0], u_data_[0] * u_std + u_mean)
                @ jnp.diag(u_std)
            )
            return jnp.mean((jac_pred - jac_true) ** 2)

        mse_total = jnp.mean(_mse(t_data, u_data))
        jac_loss_total = jnp.mean(_jac_loss(t_data, u_data))
        return mse_total + self.jacobian_weight * jac_loss_total, {
            "mse": mse_total,
            "jac_loss": jac_loss_total,
        }
