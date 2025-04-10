from collections.abc import Callable
from functools import partial
from typing import Any, Protocol

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from dynamical_systems.continuous import solve_ode
from jaxtyping import Array, Float

from .models import NeuralODE


class LossProtocol(Protocol):
    """
    A protocol defining the function signature for loss functions that are compatible
    with the training methods in this library.
    """

    # TODO: later, relax the type constraint on model
    def __call__(
        self,
        model: NeuralODE,
        t_data: Float[Array, "batch time"],
        u_data: Float[Array, "batch time dim"],
        args: Any,
    ) -> tuple[Float[Array, ""], Any]: ...


@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(None, 0, 0))
def solve_neuralode(model, t, u0):
    u_pred = solve_ode(
        model,
        t,
        u0,
        rtol=1e-4,
        atol=1e-4,
        max_steps=2048,
        adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=4096),
    )
    return u_pred


def loss_mse(
    model,
    t_data: Float[Array, "batch time"],
    u_data: Float[Array, "batch time dim"],
    args=None,
):
    del args
    u_pred = solve_neuralode(model, t_data, u_data[:, 0])
    return jnp.mean((u_pred - u_data) ** 2), None


class JacobianMatchingMSE(LossProtocol):
    analytical_jacobian: Callable
    jabobian_weight: float = 1.0

    def __call__(
        self,
        model: NeuralODE,
        t_data: Float[Array, "batch time"],
        u_data: Float[Array, "batch time dim"],
        args: Any,
    ) -> tuple[Float[Array, ""], Any]:
        pass
