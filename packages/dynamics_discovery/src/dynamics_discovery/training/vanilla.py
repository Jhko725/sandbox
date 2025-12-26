from collections.abc import Callable
from pathlib import Path
from typing import Literal

import equinox as eqx
import optax
import wandb

from dynamics_discovery.data.loaders import SegmentLoader
from dynamics_discovery.loss_functions import AbstractDynamicsLoss

from .base import BaseTrainer


class VanillaTrainer(BaseTrainer):
    optimizer: optax.GradientTransformation
    max_epochs: int
    savedir: Path
    savename: str
    logger: wandb.sdk.wandb_run.Run

    def __init__(
        self,
        optimizer: optax.GradientTransformation = optax.adabelief(1e-3),
        max_epochs: int = 5000,
        savedir: Path | str = "./results",
        savename: str = "checkpoint.eqx",
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
        wandb_mode: Literal["online", "offline", "disabled", "shared"] = "online",
    ):
        super().__init__(
            optimizer,
            max_epochs,
            savedir,
            savename,
            wandb_entity,
            wandb_project,
            wandb_mode,
        )

    def make_step_fn(
        self, loader: SegmentLoader, loss_fn: AbstractDynamicsLoss
    ) -> Callable:
        loss_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

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
