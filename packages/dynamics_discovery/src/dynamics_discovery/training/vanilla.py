from collections.abc import Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import optax
import wandb

from dynamics_discovery.data.loaders import AbstractSegmentLoader
from dynamics_discovery.io import save_model as save_model_
from dynamics_discovery.loss_functions import AbstractDynamicsLoss, MSELoss
from dynamics_discovery.models.abstract import AbstractDynamicsModel


class VanillaTrainer:
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
    ):
        self.optimizer = optimizer
        self.max_epochs = max_epochs

        self.savedir = savedir
        self.savename = savename

        self.logger = wandb.init(entity=wandb_entity, project=wandb_project)

    def train(
        self,
        model: AbstractDynamicsModel,
        loader: AbstractSegmentLoader,
        loss_fn: AbstractDynamicsLoss = MSELoss(),
        args: Any = None,
        *,
        config: dict | None = None,
        **kwargs,
    ):
        _loss_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

        @eqx.filter_jit
        def _step_fn(model_, args_, loader_state, opt_state):
            batch, loader_state_next = loader.load_batch(loader_state)
            (loss, log_dict), grads = _loss_grad_fn(model_, batch, args_, **kwargs)
            updates, opt_state_next = self.optimizer.update(
                grads, opt_state, eqx.filter(model_, eqx.is_inexact_array)
            )
            model_ = eqx.apply_updates(model_, updates)
            return loss, log_dict, model_, loader_state_next, opt_state_next

        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        loader_state = loader.init()
        with self.logger as logger:
            if config is not None:
                self.logger.config.update(config)

            loss_history = []
            for step in range(self.max_epochs):
                loss, log_dict, model, loader_state, opt_state = _step_fn(
                    model, args, loader_state, opt_state
                )
                logger.log(log_dict, step=step)
                logger.log(
                    {"train_loss": loss, "epoch": step // loader.num_batches}, step=step
                )
                print(f"{step=}, {loss=}")
                loss_history.append(loss.item())

        return model, jnp.asarray(loss_history)

    @property
    def savedir(self) -> Path:
        return self.__savedir

    @savedir.setter
    def savedir(self, value: Path | str):
        self.__savedir = Path(value)
        self.__savedir.mkdir(parents=True, exist_ok=True)

    @property
    def savepath(self) -> Path:
        return self.savedir / self.savename

    def save_model(
        self,
        model: AbstractDynamicsModel,
        config: Mapping[str, Any] | None = None,
        *,
        overwrite: bool = False,
    ):
        # Handle the case when config is not passed
        save_model_(self.savepath, model, config, overwrite)
