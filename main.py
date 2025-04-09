from functools import partial
from pathlib import Path
from collections.abc import Callable

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx
import diffrax as dfx
import optax
from dynamical_systems.dataset import TimeSeriesDataset
from dynamical_systems.continuous import solve_ode
from dynamics_discovery.preprocessing import split_into_chunks, standardize, add_noise
from dynamics_discovery.tree_utils import tree_satisfy_float_precision
import matplotlib.pyplot as plt


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
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
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

    with wandb_run:
        loss_history = []
        for epoch in range(max_epochs):
            loss, model, opt_state = make_step(
                model, t_data, u_data, u0_data, opt_state
            )
            wandb_run.log({"loss": loss, "epoch": epoch}, step=epoch)
            print(f"{epoch=}, {loss=}")
            loss_history.append(loss.item())

    return model, jnp.asarray(loss_history)


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", cfg.enable_x64)

    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config_dict.pop("wandb")

    model = hydra.utils.instantiate(cfg.model)
    dataset = TimeSeriesDataset.load(cfg.data.loadpath)

    if not tree_satisfy_float_precision(model, dataset, expect_x64=cfg.enable_x64):
        raise TypeError(
            """Model and/or dataset does not conform to the 
            expected floating point precision!"""
        )

    u_train = standardize(dataset.u[0])

    t_train_batched, u_train_batched = jax.tree.map(
        lambda x: split_into_chunks(
            x, cfg.preprocessing.batch_length, cfg.preprocessing.overlap
        ),
        (dataset.t, u_train),
    )
    u_train_batched = add_noise(
        u_train_batched,
        cfg.preprocessing.noise.rel_noise_strength,
        cfg.preprocessing.noise.preserve_first,
        cfg.preprocessing.noise.key,
    )
    optimizer_fn = hydra.utils.get_method(cfg.training.optimizer_fn)

    fig, ax = plt.subplots(1, 1)
    ax.plot(t_train_batched[0], u_train_batched[0, :, 0])
    fig.savefig("./test.png")
    run = wandb.init(
        entity=cfg.wandb.entity, project=cfg.wandb.project, config=config_dict
    )
    model = train_vanilla(
        model,
        t_train_batched,
        u_train_batched,
        optimizer_fn=optimizer_fn,
        lr=cfg.training.lr,
        max_epochs=cfg.training.max_epochs,
        wandb_run=run,
    )
    savedir = Path(cfg.checkpointing.savedir)
    if not savedir.exists():
        savedir.mkdir()
    eqx.tree_serialise_leaves(
        savedir
        / f"lorenz_length={cfg.preprocessing.batch_length}_key={cfg.model.key}.eqx",
        model,
    )


if __name__ == "__main__":
    main()

# %%
