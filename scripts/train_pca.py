import hydra
import jax
import numpy as np
from dynamics_discovery.data.dataset import TimeSeriesDataset
from dynamics_discovery.data.loaders import SegmentLoader
from dynamics_discovery.loss_functions import PushforwardMatchingMSE
from dynamics_discovery.pushforward import (
    estimate_pushforward_matrices,
)
from dynamics_discovery.training.vanilla import VanillaTrainer
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    config_path="./configs", config_name="config_neighborhood", version_base=None
)
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", cfg.enable_x64)

    model = hydra.utils.instantiate(cfg.model)
    dataset, _ = (
        TimeSeriesDataset(
            *TimeSeriesDataset.from_hdf5(cfg.data.dataset.loadpath)[::100]
        )
        .downsample(cfg.data.downsample_factor)
        .split_along_time(500)[0]
        .add_noise(cfg.data.noise_std_relative)
        .standardize()
    )
    M1, M2, scores = estimate_pushforward_matrices(
        dataset,
        cfg.neighborhood.radius,
        cfg.neighborhood.dim_project,
        cfg.neighborhood.num_neighbor_threshold,
        cfg.data.segment_length - 1,
    )
    masks = np.logical_and(scores < cfg.neighborhood.pca_score_cutoff, scores >= 0)
    loader = SegmentLoader(
        dataset,
        cfg.data.segment_length,
        hydra.utils.instantiate(cfg.data.batch_strategy),
        aux_data=(M1, M2, masks),
    )

    trainer: VanillaTrainer = hydra.utils.instantiate(cfg.training)
    trainer.savedir = trainer.savedir / f"cutoff={cfg.neighborhood.pca_score_cutoff}/"
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    model, _ = trainer.train(
        model,
        loader,
        PushforwardMatchingMSE(weight=cfg.neighborhood.weight),
        config=config_dict,
    )


if __name__ == "__main__":
    main()
