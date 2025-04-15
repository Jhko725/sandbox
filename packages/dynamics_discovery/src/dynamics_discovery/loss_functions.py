import abc
from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

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
        t_data: Float[Array, "batch time"],
        u_data: Float[Array, "batch time dim"],
        args: Any,
        **kwargs: Any,
    ) -> FloatScalar: ...


class MSELoss(AbstractDynamicsLoss):
    def __call__(
        self,
        model: AbstractDynamicsModel,
        t_data: Float[Array, " batch time"],
        u_data: Float[Array, " batch time dim"],
        args: Any = None,
        **kwargs: Any,
    ) -> FloatScalar:
        @partial(eqx.filter_vmap, in_axes=(0, 0))
        def _mse(t_data_: Float[Array, " time"], u_data_):
            u_pred = model.solve(t_data_, u_data_[0], args, **kwargs)
            return jnp.mean((u_pred - u_data_) ** 2)

        mse_batch = _mse(t_data, u_data)
        return jnp.mean(mse_batch)


class JacobianMatchingMSE(AbstractDynamicsLoss):
    analytical_jacobian: Callable = eqx.field(static=True)
    jabobian_weight: float = 1.0

    def __call__(
        self,
        model: AbstractDynamicsModel,
        t_data: Float[Array, "batch time"],
        u_data: Float[Array, "batch time dim"],
        args: Any,
        **kwargs: Any,
    ) -> FloatScalar:
        pass
