from collections.abc import Callable
from typing import Any, Literal
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PyTree
import optax
import wandb

from dynamics_discovery.data.loaders import SegmentLoader
from dynamics_discovery.loss_functions import AbstractDynamicsLoss

from .base import BaseTrainer

def normalize(x: Float[Array, "*shape"]):
    x_norm = jnp.linalg.norm(jnp.reshape(x, -1), ord=2)
    return jax.lax.cond(x_norm>0, lambda x_: x_/x_norm, jnp.zeros_like, x)

def vector_projection(x1, x2):
    """Compute the vector projection of x1 onto x2, which is the projection of
    x1 to the direction parellel to x2: proj_{x2}x_1"""
    x1, x2 = jnp.reshape(x1, -1), jnp.reshape(x2, -1)
    x2_norm = jnp.linalg.norm(x2, ord=2)
    return (jnp.dot(x1, x2)/x2_norm)*x2

def vector_rejection(x1, x2):
    """Compute the vector rejection of x1 from x2, which is the orthogonal projection of
    x1 onto the direction orthogonal to x2."""
    x1, x2 = jnp.reshape(x1, -1), jnp.reshape(x2, -1)
    return x1-vector_projection(x1, x2)


def config(grad1: Float[Array, " N"], grad2: Float[Array, " N"])-> Float[Array, " N"]:
    # TODO: implement multiterm
    u1 = normalize(vector_rejection(grad1, grad2))
    u2 = normalize(vector_rejection(grad2, grad1))
    grad_v = normalize(u1+u2)
    return (jnp.dot(grad1, grad_v)+jnp.dot(grad2, grad_v))*grad_v


def filter_value_and_grad_ConFIG(fun_multiterm: Callable[..., list[PyTree]]|Callable[..., tuple[list[PyTree], Any]], has_aux: bool = False):
    """Wraps a multi-term loss function to return both the total loss and the conflict-free gradient, as described in the ConFIG paper.
    Currently only supports two loss terms."""
    
    def _inner(*args, **kwargs):
        x, *args = args

        loss_terms, vjp_fun, *aux_list = eqx.filter_vjp(lambda _x: fun_multiterm(_x, *args, **kwargs), x, has_aux = has_aux)
        loss = sum(loss_terms)
        grads_per_loss, = jax.vmap(vjp_fun)(list(jnp.eye(len(loss_terms))))

        grads_flatten, unravel_fn = eqx.filter_vmap(jax.flatten_util.ravel_pytree, out_axes = (0, None))(grads_per_loss)
        assert grads_flatten.shape[0] == 2, "Currently only handles 2 term loss!"

        grad_config = unravel_fn(config(*grads_flatten))

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

    def __init__(
        self,
        optimizer: optax.GradientTransformation = optax.adabelief(1e-3),
        gradient_strategy: Literal["config"] = "config",
        max_epochs: int = 5000,
        savedir: Path | str = "./results",
        savename: str = "checkpoint.eqx",
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
    ):
        super().__init__(optimizer, max_epochs, savedir, savename, wandb_entity, wandb_project)
        self.gradient_strategy = gradient_strategy

    def make_step_fn(self, loader: SegmentLoader, loss_fn:AbstractDynamicsLoss)-> Callable:
        loss_grad_fn = filter_value_and_grad_ConFIG(loss_fn, has_aux=True)

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
    
