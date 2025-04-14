from collections.abc import Callable
from functools import partial
from pathlib import Path

import diffrax as dfx
import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
from dynamical_systems.dataset import TimeSeriesDataset
from dynamical_systems.metrics import lyapunov_gr
from dynamics_discovery.preprocessing import split_into_chunks
from jaxtyping import Array, Float
from omegaconf import DictConfig, OmegaConf


jax.config.update("jax_enable_x64", True)


@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(None, 0, 0))
def solve_neuralode(model, t, u0):
    lyapunov, u_pred = lyapunov_gr(
        model,
        u0,
        t,
        rtol=1e-4,
        atol=1e-4,
        max_steps=4,
        adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=4),
    )
    return u_pred, lyapunov


def loss_mse(
    model,
    t_data: Float[Array, "batch time"],
    u_data: Float[Array, "batch time dim"],
    u0_data=None,
):
    del u0_data
    u_pred, lyapunov = solve_neuralode(model, t_data, u_data[:, 0])
    return jnp.mean((u_pred - u_data) ** 2), lyapunov


def train_vanilla(
    model,
    t_data,
    u_data,
    u0_data=None,
    loss_fn=loss_mse,
    optimizer_fn: Callable = optax.adabelief,
    lr: float = 1e-3,
    max_epochs: int = 5000,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_config: dict | None = None,
):
    optimizer = optimizer_fn(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @partial(eqx.filter_value_and_grad, has_aux=True)
    def loss_grad_fn(model, t_batch, u_batch, u0_batch):
        return loss_fn(model, t_batch, u_batch, u0_batch)

    @eqx.filter_jit
    def make_step(model, t_data, u_data, u0_data, opt_state):
        (loss, lyapunov), grads = loss_grad_fn(model, t_data, u_data, u0_data)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, lyapunov

    with wandb.init(
        entity=wandb_entity, project=wandb_project, config=wandb_config
    ) as run:
        loss_history = []
        for epoch in range(max_epochs):
            loss, model, opt_state, lyapunov = make_step(
                model, t_data, u_data, u0_data, opt_state
            )

            lya_mean = jnp.mean(lyapunov, axis=0)[-1]
            print(f"{epoch=}, {loss=}")
            for i in range(len(lya_mean)):
                # run.log({f"lambda_{i}": wandb.Histogram(lyapunov[:, i])}, step=epoch)
                run.log({f"lambda_{i}_mean": lya_mean[i]}, step=epoch)
            run.log(
                {"loss": loss, "epoch": epoch, "lambda_max": jnp.max(lya_mean)},
                step=epoch,
            )
            loss_history.append(loss.item())

    return model, jnp.asarray(loss_history)


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config_dict.pop("wandb")

    model = hydra.utils.instantiate(cfg.model)
    dataset = TimeSeriesDataset.load(cfg.data.loadpath)

    # TODO: refactor the standardization part
    u_train_test = dataset.u
    u_train_mean = jnp.mean(u_train_test[0], axis=0)
    u_train_std = jnp.std(u_train_test[0], axis=0)
    u_train_test_norm = (u_train_test - u_train_mean) / u_train_std

    u_train, u_test = u_train_test_norm
    t_train_batched, _ = split_into_chunks(dataset.t, cfg.preprocessing.batch_length)
    u_train_batched, _ = split_into_chunks(u_train, cfg.preprocessing.batch_length)

    optimizer_fn = hydra.utils.get_method(cfg.training.optimizer_fn)

    model = train_vanilla(
        model,
        t_train_batched,
        u_train_batched,
        optimizer_fn=optimizer_fn,
        lr=cfg.training.lr,
        max_epochs=cfg.training.max_epochs,
        wandb_entity=cfg.wandb.entity,
        wandb_project=cfg.wandb.project,
        wandb_config=config_dict,
    )
    savedir = Path(cfg.checkpointing.savedir)
    eqx.tree_serialise_leaves(
        savedir
        / f"""lorenz_length={cfg.preprocessing.batch_length}_key={cfg.model.key}
        _lyapunov.eqx""",
        model,
    )


if __name__ == "__main__":
    main()
