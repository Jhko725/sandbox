from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import optax
import wandb

from ..io import save_model as save_model_
from ..loss_functions import AbstractDynamicsLoss, MSELoss
from ..models.abstract import AbstractDynamicsModel


class VanillaTrainer:
    optimizer: optax.GradientTransformation
    max_epochs: int
    savepath: Path
    logger: wandb.sdk.wandb_run.Run

    def __init__(
        self,
        optimizer: optax.GradientTransformation = optax.adabelief(1e-3),
        loss_fn: AbstractDynamicsLoss = MSELoss(),
        max_epochs: int = 5000,
        savedir: Path | str = "./results",
        savename: str = "checkpoint.eqx",
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
    ):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs

        savedir = Path(savedir)
        savedir.mkdir(parents=True, exist_ok=True)
        self.savepath = Path(savedir) / savename

        self.logger = wandb.init(entity=wandb_entity, project=wandb_project)

    def make_step_fn(self) -> Callable:
        loss_grad_fn = eqx.filter_value_and_grad(self.loss_fn)

        @eqx.filter_jit
        def _step(model, t_data, u_data, args, opt_state):
            loss, grads = loss_grad_fn(model, t_data, u_data, args)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state

        return _step

    def train(
        self,
        model: AbstractDynamicsModel,
        t_train_batched,
        u_train_batched,
        *,
        config: dict | None = None,
    ):
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        step_fn = self.make_step_fn()

        with self.logger as logger:
            if config is not None:
                self.logger.config.update(config)

            loss_history = []
            for epoch in range(self.max_epochs):
                loss, model, opt_state = step_fn(
                    model, t_train_batched, u_train_batched, None, opt_state
                )
                logger.log({"loss": loss, "epoch": epoch}, step=epoch)
                print(f"{epoch=}, {loss=}")
                loss_history.append(loss.item())

        return model, jnp.asarray(loss_history)

    def save_model(
        self,
        model: AbstractDynamicsModel,
        config: Mapping[str, Any] | None = None,
        *,
        overwrite: bool = False,
    ):
        # Handle the case when config is not passed
        save_model_(self.savepath, model, config, overwrite)
