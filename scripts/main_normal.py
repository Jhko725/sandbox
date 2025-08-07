from functools import partial
from typing import Any

import diffrax as dfx
import hydra
import jax
import jax.numpy as jnp
from dynamical_systems.continuous import AbstractODE
from dynamics_discovery.custom_types import FloatScalar
from dynamics_discovery.data.dataset import TimeSeriesDataset
from dynamics_discovery.data.loaders import AbstractBatching
from dynamics_discovery.loss_functions import AbstractDynamicsLoss
from dynamics_discovery.models.abstract import AbstractDynamicsModel
from dynamics_discovery.neighborhood import NeighborhoodSegmentLoader

from dynamics_discovery.training.vanilla import VanillaTrainer
from jaxtyping import Array, Float, PyTree, PRNGKeyArray
from omegaconf import DictConfig, OmegaConf
from ott.utils import batched_vmap


class NormalODE(AbstractODE):
    ode: AbstractODE

    @property
    def dim(self) -> int:
        return self.ode.dim * (self.ode.dim + 1)

    @property
    def solver(self) -> dfx.AbstractSolver:
        return self.ode.solver

    @property
    def dt0(self) -> float | None:
        return self.ode.dt0

    @property
    def stepsize_controller(self) -> dfx.AbstractStepSizeController:
        return self.ode.stepsize_controller

    def rhs(self, t, u: tuple[Float[Array, " dim"], Float[Array, " dim"]], args):
        x, Nx = u

        def rhs_ode(x_):
            return self.ode.rhs(t, x_, args)

        dx, vjp_fun = jax.vjp(rhs_ode, x)
        dNx = -vjp_fun(Nx)[0]

        return dx, dNx

    def solve(
        self,
        ts: Float[Array, " time"],
        u0: tuple[Float[Array, " dim"], Float[Array, "dim"]],
        args=None,
        **kwargs,
    ) -> tuple[Float[Array, " time dim"], Float[Array, " time dim"]]:
        sol = dfx.diffeqsolve(
            dfx.ODETerm(self.rhs),
            self.solver,
            ts[0],
            ts[-1],
            self.dt0,
            u0,
            args,
            saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=self.stepsize_controller,
            **kwargs,
        )
        return sol.ys


def get_normal_direction(points: Float[Array, "batch dim"]):
    points_centered = points  # - jnp.mean(points, axis=0)
    _, _, Q_T = jnp.linalg.svd(points_centered, full_matrices=False)
    return Q_T[2]


get_normal_direction_batch = jax.vmap(jax.vmap(get_normal_direction))


class NormalVectorSegmentLoader(NeighborhoodSegmentLoader):
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        segment_length: int,
        num_neighbors: int,
        batch_strategy: AbstractBatching,
    ):
        super().__init__(dataset, segment_length, num_neighbors, batch_strategy)

    def load_batch(
        self, loader_state: PRNGKeyArray
    ) -> tuple[
        Float[Array, "{self.batch_size} {self.segment_length}"],
        Float[Array, "{self.batch_size} {self.segment_length} dim"],
        Float[
            Array, "{self.batch_size} {self.segment_length} {self.num_neighbors} dim"
        ],
    ]:
        (t_batch, u_batch, u_nn_batch), loader_state_next = super().load_batch(
            loader_state
        )
        normal_batch = get_normal_direction_batch(u_nn_batch)
        return (t_batch, u_batch, normal_batch), loader_state_next


def directional_cosine_squared(x1, x2):
    denom = jnp.sum(x1 * x1) * jnp.sum(x2 * x2)
    return jnp.dot(x1, x2) ** 2 / denom


class NormalLoss(AbstractDynamicsLoss):
    weight: float = 1.0
    batch_size: int | None = None

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
            sin_sqr = 1 - jax.vmap(directional_cosine_squared)(Nu_data_, Nu_pred)
            ortho = jax.vmap(directional_cosine_squared)(
                jax.vmap(lambda t_, u_: model.ode.rhs(t_, u_, None))(t_data_, u_data_),
                Nu_data_,
            )
            return mse_u, jnp.mean(sin_sqr), jnp.mean(ortho)

        mse_, sin_sqr_, ortho_ = _loss(t_data, u_data, Nu_data)
        mse_total = jnp.mean(mse_)
        sin_sqr_total = jnp.mean(sin_sqr_)
        ortho_total = jnp.mean(ortho_)
        return mse_total + self.weight * sin_sqr_total + self.weight * ortho_total, {
            "mse": mse_total,
            "normal_loss": sin_sqr_total,
            "ortho_loss": ortho_total,
        }


@hydra.main(
    config_path="./configs", config_name="config_neighborhood", version_base=None
)
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", cfg.enable_x64)

    model = NormalODE(
        hydra.utils.instantiate(cfg.model),
    )
    dataset, _ = (
        TimeSeriesDataset.from_hdf5(cfg.data.dataset.loadpath)
        .downsample(cfg.data.downsample_factor)
        .add_noise(cfg.data.noise_std_relative)
        .standardize()
    )
    loader = NormalVectorSegmentLoader(
        dataset,
        cfg.data.segment_length,
        cfg.neighborhood.num_neighbors,
        hydra.utils.instantiate(cfg.data.batch_strategy),
    )

    trainer: VanillaTrainer = hydra.utils.instantiate(cfg.training)
    trainer.savedir = trainer.savedir / f"normal={cfg.neighborhood.weight}"
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    loss_fn = NormalLoss(
        cfg.neighborhood.weight,
        cfg.neighborhood.chunk_size,
    )
    model, _ = trainer.train(model, loader, loss_fn, config=config_dict)
    trainer.save_model(model, config_dict["model"])


if __name__ == "__main__":
    main()
