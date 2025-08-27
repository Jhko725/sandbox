from functools import partial
from typing import Any

import diffrax as dfx
import hydra
import jax
import jax.numpy as jnp
from dynamical_systems.continuous import AbstractODE, TangentODE
from dynamical_systems.transforms import TransformedODE
from dynamics_discovery.custom_types import FloatScalar
from dynamics_discovery.data.dataset import TimeSeriesDataset
from dynamics_discovery.data.loaders import SegmentLoader
from dynamics_discovery.loss_functions import AbstractDynamicsLoss
from dynamics_discovery.models.abstract import AbstractDynamicsModel
from dynamics_discovery.training.vanilla import VanillaTrainer
from jaxtyping import Array, Float, PyTree
from omegaconf import DictConfig, OmegaConf
from ott.utils import batched_vmap


def tangent_evolution_matrix(
    ode: AbstractODE,
    x: Float[Array, " dim"],
    t0: float,
    t1: float,
    solver: dfx.AbstractAdaptiveSolver = dfx.Tsit5(),
    stepsize_controller: dfx.AbstractAdaptiveStepSizeController = dfx.PIDController(
        rtol=1e-4, atol=1e-6
    ),
):
    tangent_ode = TangentODE(ode)
    u0 = (x, jnp.identity(ode.dim))

    sol = dfx.diffeqsolve(
        dfx.ODETerm(tangent_ode.rhs),
        solver,
        t0,
        t1,
        None,
        u0,
        None,
        saveat=dfx.SaveAt(t1=True),
        stepsize_controller=stepsize_controller,
    )
    _, M_t = sol.ys
    return M_t[0]  # remove time dimension


class TangentEvolutionMatchingMSE(AbstractDynamicsLoss):
    ode_true: AbstractODE
    weight: float
    batch_size: int | None

    def __init__(
        self,
        ode_true: AbstractODE,
        weight: float = 1.0,
        batch_size: int | None = None,
    ):
        self.ode_true = ode_true
        self.weight = weight
        self.batch_size = batch_size

    def __call__(
        self,
        model: AbstractDynamicsModel,
        batch: PyTree[Float[Array, "batch ..."]],
        args: Any = None,
        **kwargs: Any,
    ) -> FloatScalar:
        t_data, u_data = batch

        batch_size = u_data.shape[0] if self.batch_size is None else self.batch_size

        @partial(batched_vmap, in_axes=(0, 0), batch_size=batch_size)
        def _mse(t_data_: Float[Array, " time"], u_data_):
            u_pred = model.solve(t_data_, u_data_[0], args, **kwargs)
            return jnp.mean((u_pred - u_data_) ** 2)

        @partial(batched_vmap, in_axes=(0, 0), batch_size=batch_size)
        def _tangent_evol_loss(t_data_: Float[Array, " time"], u_data_):
            evol_pred = tangent_evolution_matrix(
                model,
                u_data_[0],
                t_data_[0],
                t_data_[-1],
                model.solver,
                model.stepsize_controller,
            )
            evol_true = tangent_evolution_matrix(
                self.ode_true,
                u_data_[0],
                t_data_[0],
                t_data_[-1],
                model.solver,
                model.stepsize_controller,
            )
            return jnp.mean((evol_pred - evol_true) ** 2)

        mse_total = jnp.mean(_mse(t_data, u_data))
        evol_loss_total = jnp.mean(_tangent_evol_loss(t_data, u_data))
        return mse_total + self.weight * evol_loss_total, {
            "mse": mse_total,
            "tangent_evolution_loss": evol_loss_total,
        }


@hydra.main(config_path="./configs", config_name="config_jacobian", version_base=None)
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", cfg.enable_x64)

    model = hydra.utils.instantiate(cfg.model)
    dataset, transform = (
        TimeSeriesDataset.from_hdf5(cfg.data.dataset.loadpath)
        .downsample(cfg.data.downsample_factor)
        .add_noise(cfg.data.noise_std_relative)
        .standardize()
    )

    loader = SegmentLoader(
        dataset,
        cfg.data.segment_length,
        hydra.utils.instantiate(cfg.data.batch_strategy),
    )

    trainer: VanillaTrainer = hydra.utils.instantiate(cfg.training)
    trainer.savedir = trainer.savedir / "tangent_evolution/"
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    ode_true = hydra.utils.instantiate(cfg.data.dataset.ode)
    loss_fn = TangentEvolutionMatchingMSE(
        TransformedODE(ode_true, transform),
        cfg.jacobian.weight,
        cfg.jacobian.chunk_size,
    )
    model, _ = trainer.train(model, loader, loss_fn, config=config_dict)
    trainer.save_model(model, config_dict["model"])


if __name__ == "__main__":
    main()
