from functools import partial
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import diffrax as dfx
from dynamical_systems.continuous import AbstractODE
from dynamics_discovery.custom_types import FloatScalar
from dynamics_discovery.dataset import TimeSeriesDataset
from dynamics_discovery.loss_functions import AbstractDynamicsLoss
from dynamics_discovery.models.abstract import AbstractDynamicsModel
from dynamics_discovery.models.neighborhood import (
    create_neighborhood_dataset,
)
from dynamics_discovery.preprocessing import (
    add_noise,
    downsample,
    standardize,
)
from dynamics_discovery.training.vanilla import VanillaTrainer
from dynamics_discovery.utils.tree import tree_satisfy_float_precision
from jaxtyping import Array, Float, PyTree
from omegaconf import DictConfig, OmegaConf
from ott.utils import batched_vmap


class NormalODE(AbstractODE):
    ode: AbstractODE

    @property
    def dim(self) -> int:
        return self.ode.dim * (self.ode.dim + 1)

    def rhs(self, t, u: tuple[Float[Array, " dim"], Float[Array, " dim"]], args):
        x, Nx = u

        def rhs_ode(x_):
            return self.ode.rhs(t, x_, args)

        dx, vjp_fun = jax.vjp(rhs_ode, x)
        dNx = -vjp_fun(Nx)[0]

        return dx, dNx

    @property
    def default_solver(self):
        return self.ode.solver

    @property
    def default_atol(self):
        return self.ode.default_atol

    @property
    def default_rtol(self):
        return self.ode.default_rtol

    def solve(
        self,
        ts: Float[Array, " time"],
        u0: tuple[Float[Array, " dim"], Float[Array, " dim"]],
        args=None,
        **kwargs,
    ) -> tuple[Float[Array, " time dim"], Float[Array, " time dim"]]:
        sol = dfx.diffeqsolve(
            dfx.ODETerm(self.rhs),
            self.ode.solver,
            ts[0],
            ts[-1],
            self.ode.dt0,
            u0,
            args,
            saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=self.ode.stepsize_controller,
            **kwargs,
        )
        return sol.ys


def get_normal_direction(points: Float[Array, "batch dim"]):
    points_centered = points - jnp.mean(points, axis=0)
    _, _, Q_T = jnp.linalg.svd(points_centered)
    return Q_T[2]


def directional_cosine_squared(x1, x2):
    denom = jnp.sum(x1 * x1) * jnp.sum(x2 * x2)
    return jnp.dot(x1, x2) ** 2 / denom


class NormalLoss(AbstractDynamicsLoss):
    batch_size: int | None = None
    weight: float = 1.0

    def __call__(
        self,
        model: AbstractDynamicsModel,
        batch: PyTree[Float[Array, "batch ..."]],
        args: Any = None,
        **kwargs: Any,
    ) -> FloatScalar:
        t_data, u_data, Nu_data = batch
        batch_size = u_data.shape[0] if self.batch_size is None else self.batch_size

        @partial(batched_vmap, in_axes=(0, 0, 0), batch_size=batch_size)
        def _loss(t_data_: Float[Array, " time"], u_data_, Nu_data_):
            u_pred, Nu_pred = model.solve(
                t_data_, (u_data_[0], Nu_data_[0]), args, **kwargs
            )
            mse_u = jnp.mean((u_pred - u_data_) ** 2)
            cos_sqr = jax.vmap(directional_cosine_squared)(Nu_data_, Nu_pred)
            return mse_u, 1 - cos_sqr

        mse_, sin_sqr_ = _loss(t_data, u_data, Nu_data)
        mse_total = jnp.mean(mse_)
        sin_sqr_total = jnp.mean(sin_sqr_)
        return mse_total + self.weight * sin_sqr_total, {
            "mse": mse_total,
            "normal_loss": sin_sqr_total,
        }


@hydra.main(
    config_path="./configs", config_name="config_neighborhood", version_base=None
)
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", cfg.enable_x64)
    model = NormalODE(
        hydra.utils.instantiate(cfg.model),
    )
    dataset = TimeSeriesDataset.load(cfg.data.loadpath)

    if not tree_satisfy_float_precision(model, dataset, expect_x64=cfg.enable_x64):
        raise TypeError(
            """Model and/or dataset does not conform to the 
            expected floating point precision!"""
        )

    t_train = dataset.t[0][:: cfg.preprocessing.downsample.keep_every]
    u_train = dataset.u[0]
    u_train = add_noise(
        u_train,
        cfg.preprocessing.noise.rel_noise_strength,
        cfg.preprocessing.noise.preserve_first,
        cfg.preprocessing.noise.key,
    )
    u_train, _ = standardize(u_train)

    t, u, du, _ = create_neighborhood_dataset(
        t_train,
        u_train,
        num_neighbors=cfg.neighborhood.num_neighbors,
        train_length=cfg.preprocessing.batch_length,
        train_length_neighbors=cfg.preprocessing.batch_length,
        max_radius=cfg.neighborhood.max_radius,
        min_radius=0.0,
        adjacent_exclusion_threshold=cfg.neighborhood.adjacent_exclusion_threshold,
    )
    del dataset

    u_neigh = jnp.expand_dims(u, axis=-2) + du
    u_samples_batch = jnp.concatenate((jnp.expand_dims(u, axis=-2), u_neigh), axis=-2)
    normals = jax.vmap(jax.vmap(get_normal_direction))(u_samples_batch)

    batch = (t, u, normals)

    trainer = VanillaTrainer(
        hydra.utils.instantiate(cfg.training.optimizer),
        NormalLoss(cfg.training.loss_fn.batch_size, cfg.training.loss_fn.weight),
        cfg.training.max_epochs,
        cfg.training.savedir,
        cfg.training.savename,
        cfg.training.wandb_entity,
        cfg.training.wandb_project,
    )
    trainer.savedir = trainer.savedir / "normal/"
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    model, _ = trainer.train(model, batch, config=config_dict, max_steps=1024)
    trainer.save_model(model, config_dict["model"])


if __name__ == "__main__":
    main()
