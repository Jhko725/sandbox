from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from orbax.checkpoint._src.checkpoint_managers.preservation_policy import BestN

from dynamics_discovery.data.loaders import SegmentLoader
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
        wandb_mode: Literal["online", "offline", "disabled", "shared"] = "online",
    ):
        self.optimizer = optimizer
        self.max_epochs = max_epochs

        self.savedir = Path(savedir)
        self.savename = savename

        self.logger = wandb.init(
            entity=wandb_entity, project=wandb_project, mode=wandb_mode
        )

    def train(
        self,
        model: AbstractDynamicsModel,
        loader: SegmentLoader,
        loss_fn: AbstractDynamicsLoss,
        args: Any = None,
        metric_fns: dict[str, Callable] | None = None,
        *,
        config: dict | None = None,
        **kwargs,
    ):
        print(loss_fn)
        print(eqx.filter_value_and_grad(loss_fn, has_aux=True))
        step_fn = self.make_step_fn(loader, loss_fn, metric_fns)

        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        loader_state = loader.init()

        ckpt_manager = ocp.CheckpointManager(
            (self.savedir / self.savename).resolve(),
            options=ocp.CheckpointManagerOptions(
                preservation_policy=BestN(
                    get_metric_fn=lambda metrics: metrics["train_loss"],
                    reverse=True,
                    n=1,
                )
            ),
            metadata=config["model"],
        )

        with self.logger as logger:
            if config is not None:
                self.logger.config.update(config)

            with ckpt_manager as mngr:
                loss_history = []
                for step in range(self.max_epochs):
                    loss, log_dict, model_next, loader_state, opt_state = step_fn(
                        model, args, loader_state, opt_state
                    )
                    metrics = log_dict | {
                        "train_loss": loss,
                        "epoch": step // loader.num_batches,
                    }
                    logger.log(metrics, step=step)

                    print(f"{step=}, {loss=}")
                    loss_history.append(loss.item())
                    mngr.save(
                        step,
                        args=ocp.args.StandardSave(eqx.filter(model, eqx.is_array)),
                        metrics=metrics,
                    )
                    model = model_next
        return model, np.asarray(loss_history)

    @abstractmethod
    def make_step_fn(
        self,
        loader: SegmentLoader,
        loss_fn: AbstractDynamicsLoss,
        metric_fn_dict: dict[str, Callable] | None,
    ) -> Callable: ...

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
