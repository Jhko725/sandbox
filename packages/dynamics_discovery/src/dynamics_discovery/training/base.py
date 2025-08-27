from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np
import optax
import wandb

from dynamics_discovery.data.loaders import SegmentLoader
from dynamics_discovery.io import save_model as save_model_
from dynamics_discovery.loss_functions import AbstractDynamicsLoss
from dynamics_discovery.models.abstract import AbstractDynamicsModel


class BaseTrainer(ABC):
    optimizer: optax.GradientTransformation
    max_epochs: int
    savedir: Path
    savename: str
    logger: wandb.sdk.wandb_run.Run

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        max_epochs: int,
        savedir: Path | str,
        savename: str,
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
        loader: SegmentLoader,
        loss_fn: AbstractDynamicsLoss,
        args: Any = None,
        *,
        config: dict | None = None,
        **kwargs,
    ):
        step_fn = self.make_step_fn(loader, loss_fn)

        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        loader_state = loader.init()
        with self.logger as logger:
            if config is not None:
                self.logger.config.update(config)

            loss_history = []
            for step in range(self.max_epochs):
                loss, log_dict, model, loader_state, opt_state = step_fn(
                    model, args, loader_state, opt_state
                )
                logger.log(log_dict, step=step)
                logger.log(
                    {"train_loss": loss, "epoch": step // loader.num_batches}, step=step
                )
                print(f"{step=}, {loss=}")
                loss_history.append(loss.item())

        return model, np.asarray(loss_history)

    @abstractmethod
    def make_step_fn(self, loss_fn: AbstractDynamicsLoss) -> Callable: ...

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
