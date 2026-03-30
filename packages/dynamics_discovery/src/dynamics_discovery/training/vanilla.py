from collections.abc import Callable
from pathlib import Path
from typing import Literal

import equinox as eqx
import optax
import wandb
from jaxtyping import Array

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
        self,
        loss_fn: AbstractDynamicsLoss,
        train_loader: SegmentLoader,
        validation_loader: SegmentLoader | None,
        metric_fn_dict: dict[str, Callable] | None,
    ) -> Callable:
        loss_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

        ## TODO: find better place to place this function; probably will need to
        # refactor class hierarchy
        if metric_fn_dict is None:

            def metric_fn(model_, batch, args_) -> dict[str, Array]:
                return dict()
        else:

            def metric_fn(model_, batch, args_) -> dict[str, Array]:
                return {
                    name: fn(model_, batch, args_)
                    for name, fn in metric_fn_dict.items()
                }

        @eqx.filter_jit
        def _step_fn(model_, args_, loader_states, opt_state):
            train_state, valid_state = loader_states
            # Train dataset
            batch_train, train_state = train_loader.load_batch(train_state)

            (loss, log_dict), grads = loss_grad_fn(model_, batch_train, args_)
            updates, opt_state = self.optimizer.update(
                grads, opt_state, eqx.filter(model_, eqx.is_inexact_array)
            )

            metrics_train = {
                f"{k}_train": v
                for k, v in metric_fn(model_, batch_train, args_).items()
            }
            log_dict = {f"{k}_train": v for k, v in log_dict.items()}
            log_dict = log_dict | metrics_train | {"loss_train": loss}
            # Validation dataset
            if validation_loader is not None:
                batch_valid, valid_state = validation_loader.load_batch(valid_state)
                loss_valid, log_dict_valid = loss_fn(model_, batch_valid, args_)
                metrics_valid = {
                    f"{k}_validation": v
                    for k, v in metric_fn(model_, batch_valid, args_).items()
                }
                log_dict_valid = {
                    f"{k}_validation": v for k, v in log_dict_valid.items()
                }
                log_dict_valid = (
                    log_dict_valid | metrics_valid | {"loss_validation": loss_valid}
                )
                log_dict = log_dict | log_dict_valid
            model_ = eqx.apply_updates(model_, updates)
            return loss, log_dict, model_, (train_state, valid_state), opt_state

        return _step_fn
