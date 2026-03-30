import hydra
import jax
from dynamics_discovery.data.dataset import TimeSeriesDataset
from dynamics_discovery.neighborhood import (
    NormalLoss,
    NormalVectorSegmentLoader,
)
from dynamics_discovery.training.multiterm import MultitermTrainer
from dynamics_discovery.training.vanilla import BaseTrainer
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    config_path="./configs", config_name="config_neighborhood", version_base=None
)
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", cfg.enable_x64)

    model = hydra.utils.instantiate(cfg.model)

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

    trainer: BaseTrainer = hydra.utils.instantiate(cfg.training)
    trainer.savedir = trainer.savedir / f"normal/weight={cfg.neighborhood.weight}"
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    if isinstance(trainer, MultitermTrainer):
        multiterm = True
        # trainer.gradient_weights = [
        #     1.0,
        #     cfg.neighborhood.weight,
        #     cfg.neighborhood.weight,
        # ]
    else:
        multiterm = False

    loss_fn = NormalLoss(
        cfg.neighborhood.weight,
        cfg.neighborhood.chunk_size,
        multiterm=multiterm,
        orthogonal_loss=False,
    )
    model, _ = trainer.train(model, loader, loss_fn, config=config_dict)


if __name__ == "__main__":
    main()
