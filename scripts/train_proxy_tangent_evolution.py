import hydra
import jax
from dynamical_systems.transforms import TransformedODE
from dynamics_discovery.data.dataset import TimeSeriesDataset
from dynamics_discovery.data.loaders import SegmentLoader
from dynamics_discovery.loss_functions import TangentEvolutionMatchingMSE2
from dynamics_discovery.training.multiterm import MultitermTrainer
from dynamics_discovery.training.vanilla import BaseTrainer
from omegaconf import DictConfig, OmegaConf


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

    trainer: BaseTrainer = hydra.utils.instantiate(cfg.training)
    trainer.savedir = (
        trainer.savedir / f"proxy_tangent_evolution/weight={cfg.jacobian.weight}"
    )
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    ode_true = hydra.utils.instantiate(cfg.data.dataset.ode)
    if isinstance(trainer, MultitermTrainer):
        multiterm = True
    else:
        multiterm = False
    loss_fn = TangentEvolutionMatchingMSE2(
        TransformedODE(ode_true, transform),
        cfg.jacobian.weight,
        cfg.jacobian.chunk_size,
        multiterm=multiterm,
    )

    model, _ = trainer.train(model, loader, loss_fn, config=config_dict)


if __name__ == "__main__":
    main()
