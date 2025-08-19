import hydra
import jax
from dynamics_discovery.data.dataset import TimeSeriesDataset
from dynamics_discovery.neighborhood import (
    NeighborhoodMSELoss,
    NeighborhoodSegmentLoader,
    NeuralNeighborhoodFlow,
)
from dynamics_discovery.training.vanilla import VanillaTrainer
from dynamics_discovery.training.multiterm import MultitermTrainer
from dynamics_discovery.training.base import BaseTrainer
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    config_path="./configs", config_name="config_neighborhood", version_base=None
)
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", cfg.enable_x64)

    model = NeuralNeighborhoodFlow(
        hydra.utils.instantiate(cfg.model),
        cfg.neighborhood.second_order,
        cfg.neighborhood.use_taylor_mode,
    )
    dataset, _ = (
        TimeSeriesDataset.from_hdf5(cfg.data.dataset.loadpath)
        .downsample(cfg.data.downsample_factor)
        .add_noise(cfg.data.noise_std_relative)
        .standardize()
    )
    loader = NeighborhoodSegmentLoader(
        dataset,
        cfg.data.segment_length,
        cfg.neighborhood.num_neighbors,
        hydra.utils.instantiate(cfg.data.batch_strategy),
    )

    trainer: BaseTrainer = hydra.utils.instantiate(cfg.training)
    trainer.savedir = trainer.savedir / f"neighborhood/weight={cfg.neighborhood.weight}"
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    multiterm = True if isinstance(trainer, MultitermTrainer) else False

    loss_fn = NeighborhoodMSELoss(
        cfg.neighborhood.weight,
        cfg.neighborhood.chunk_size,
        multiterm=multiterm
    )
    model, _ = trainer.train(model, loader, loss_fn, config=config_dict)
    trainer.save_model(model, config_dict["model"])


if __name__ == "__main__":
    main()
