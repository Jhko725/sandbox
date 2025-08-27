from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import wandb
from jaxtyping import Array, ArrayLike, Float, PyTree

from dynamics_discovery.data.loaders import SegmentLoader
from dynamics_discovery.loss_functions import AbstractDynamicsLoss

from .base import BaseTrainer


def vector_normalize(x: Float[Array, " N"]) -> Float[Array, " N"]:
    assert x.ndim == 1, (
        f"Expected input to be a vector, instead given an array with shape {x.shape}"
    )
    x_norm = jnp.linalg.norm(x, ord=2)
    return jax.lax.cond(x_norm > 0, lambda x_: x_ / x_norm, jnp.zeros_like, x)


def vector_projection(
    x1: Float[Array, " N"], x2: Float[Array, " N"]
) -> Float[Array, " N"]:
    """Compute the vector projection of x1 onto x2, which is the projection of
    x1 to the direction parallel to x2: proj_{x2}x_1"""
    assert (
        x1.ndim == x2.ndim == 1
    ), f"""Expected inputs to be vectors, instead got arrays with shapes
          {x1.shape}, {x2.shape}"""
    x2_norm = jnp.linalg.norm(x2, ord=2)
    return (jnp.dot(x1, x2) / x2_norm) * x2


def vector_rejection(
    x1: Float[Array, " N"], x2: Float[Array, " N"]
) -> Float[Array, " N"]:
    """Compute the vector rejection of x1 from x2, which is the orthogonal projection of
    x1 onto the direction orthogonal to x2."""
    return x1 - vector_projection(x1, x2)


def gradient_ConFIG(
    per_term_grads: Float[Array, "terms params"],
    weights: Float[Array, " terms"] | None = None,
) -> Float[Array, " params"]:
    if weights is None:
        weights = jnp.ones(per_term_grads.shape[0])

    if per_term_grads.shape[0] == 2:
        return _gradient_ConFIG_twoterm(per_term_grads, weights)
    else:
        return _gradient_ConFIG_multiterm(per_term_grads, weights)


def _gradient_ConFIG_multiterm(
    per_term_grads: Float[Array, "terms params"],
    weights: Float[Array, " terms"],
) -> Float[Array, " params"]:
    grads_normed = jax.vmap(vector_normalize, in_axes=0)(per_term_grads)
    grad_u: Float[Array, " params"] = vector_normalize(
        jnp.linalg.pinv(grads_normed) @ weights
    )
    return jnp.sum(per_term_grads @ grad_u) * grad_u


def _gradient_ConFIG_twoterm(
    per_term_grads: Float[Array, "2 params"],
    weights: Float[Array, " 2"],
) -> Float[Array, " params"]:
    grad1, grad2 = per_term_grads
    u1 = vector_normalize(vector_rejection(grad2, grad1))
    u2 = vector_normalize(vector_rejection(grad1, grad2))
    grad_v: Float[Array, " params"] = vector_normalize(
        u1 * weights[0] + u2 * weights[1]
    )
    return jnp.sum(per_term_grads @ grad_v) * grad_v


def filter_value_and_grad_ConFIG(
    fun_multiterm: Callable[..., list[PyTree]]
    | Callable[..., tuple[list[PyTree], Any]],
    weights: Float[ArrayLike, " terms"] | None = None,
    has_aux: bool = False,
):
    """Wraps a multi-term loss function to return both the total loss and the conflict-
    free gradient, as described in the ConFIG paper."""
    if weights is not None:
        weights = jnp.asarray(weights)

    def _inner(*args, **kwargs):
        x, *args = args

        loss_terms, vjp_fun, *aux_list = eqx.filter_vjp(
            lambda _x: fun_multiterm(_x, *args, **kwargs), x, has_aux=has_aux
        )
        loss = sum(loss_terms)
        (grads_per_loss,) = jax.vmap(vjp_fun)(list(jnp.eye(len(loss_terms))))

        grads_flatten, unravel_fn = eqx.filter_vmap(
            jax.flatten_util.ravel_pytree, out_axes=(0, None)
        )(grads_per_loss)

        grad_config = unravel_fn(gradient_ConFIG(grads_flatten, weights))

        if has_aux:
            return (loss, aux_list[0]), grad_config
        else:
            return loss, grad_config

    return _inner


class MultitermTrainer(BaseTrainer):
    optimizer: optax.GradientTransformation
    max_epochs: int
    savedir: Path
    savename: str
    logger: wandb.sdk.wandb_run.Run
    gradient_strategy: str
    gradient_weights: Float[Array, " num_terms"]

    def __init__(
        self,
        optimizer: optax.GradientTransformation = optax.adabelief(1e-3),
        gradient_strategy: Literal["config"] = "config",
        gradient_weights: Sequence[float] = [1.0, 1.0, 1.0],
        max_epochs: int = 5000,
        savedir: Path | str = "./results",
        savename: str = "checkpoint.eqx",
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
    ):
        super().__init__(
            optimizer, max_epochs, savedir, savename, wandb_entity, wandb_project
        )
        self.gradient_strategy = gradient_strategy
        self.gradient_weights = jnp.asarray(gradient_weights)

    def make_step_fn(
        self, loader: SegmentLoader, loss_fn: AbstractDynamicsLoss
    ) -> Callable:
        loss_grad_fn = filter_value_and_grad_ConFIG(
            loss_fn, self.gradient_weights, has_aux=True
        )

        @eqx.filter_jit
        def _step_fn(model_, args_, loader_state, opt_state):
            batch, loader_state_next = loader.load_batch(loader_state)
            (loss, log_dict), grads = loss_grad_fn(model_, batch, args_)
            updates, opt_state_next = self.optimizer.update(
                grads, opt_state, eqx.filter(model_, eqx.is_inexact_array)
            )
            model_ = eqx.apply_updates(model_, updates)
            return loss, log_dict, model_, loader_state_next, opt_state_next

        return _step_fn
