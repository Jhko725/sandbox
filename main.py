from functools import partial
from pathlib import Path
from collections.abc import Callable

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx
import diffrax as dfx
import optax
from dynamical_systems.dataset import TimeSeriesDataset
from dynamical_systems.continuous import solve_ode


# TODO: move this to dynamical_systems.dataset
# TODO: add a share_time: bool parameter
def split_into_chunks(
    sequence: Float[Array, " N"], chunk_size: int
) -> tuple[Float[Array, "B N"], Float[Array, " N_remainder"] | None]:
    # TODO: Handle batch dimension in the sequence argument
    # TODO: Implement the case when there are overlaps between chunks, as specified by the overlap: int parameter
    chunks = jnp.split(sequence, jnp.arange(chunk_size, len(sequence), chunk_size))
    if len(chunks[-2]) == len(chunks[-1]):
        batched_chunks = jnp.stack(chunks, axis=0)
        remainder = None
    else:
        batched_chunks = jnp.stack(chunks[:-1], axis=0)
        remainder = chunks[-1]
    return batched_chunks, remainder


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
    u0_data=None,
):
    del u0_data
    u_pred = solve_neuralode(model, t_data, u_data[:, 0])
    return jnp.mean((u_pred - u_data) ** 2)


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

    @eqx.filter_value_and_grad
    def loss_grad_fn(model, t_batch, u_batch, u0_batch):
        return loss_fn(model, t_batch, u_batch, u0_batch)

    @eqx.filter_jit
    def make_step(model, t_data, u_data, u0_data, opt_state):
        loss, grads = loss_grad_fn(model, t_data, u_data, u0_data)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    with wandb.init(
        entity=wandb_entity, project=wandb_project, config=wandb_config
    ) as run:
        loss_history = []
        for epoch in range(max_epochs):
            loss, model, opt_state = make_step(
                model, t_data, u_data, u0_data, opt_state
            )
            run.log({"loss": loss, "epoch": epoch}, step=epoch)
            print(f"{epoch=}, {loss=}")
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
        savedir / f"lorenz_length={cfg.preprocessing.batch_length}.eqx"
    )


if __name__ == "__main__":
    main()
