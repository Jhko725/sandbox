import hydra
import jax
from dynamics_discovery.data.dataset import TimeSeriesDataset
from dynamics_discovery.neighborhood.neighborhood import (
    NeighborhoodMSELoss2,
    NeighborhoodSegmentLoader,
)
from dynamics_discovery.neighborhood.radius_selection import find_r_max, find_r_min
from dynamics_discovery.training.base import BaseTrainer
from dynamics_discovery.training.multiterm import MultitermTrainer
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    config_path="./configs", config_name="config_neighborhood", version_base=None
)
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", cfg.enable_x64)

    # Manually select gpu to run on
    if cfg.gpu_id != "auto":
        jax.config.update("jax_default_device", jax.devices("gpu")[cfg.gpu_id])

    model = hydra.utils.instantiate(cfg.model)
    dataset, _ = (
        TimeSeriesDataset(
            *TimeSeriesDataset.from_hdf5(cfg.data.dataset.loadpath)[::100]
        )
        .downsample(cfg.data.downsample_factor)
        .split_along_time(500)[0]
        .standardize()
    )
    dataset = dataset.add_noise(cfg.data.noise_std_relative)

    r_min = find_r_min(
        dataset, max_allowed_rel_noise=0.4, multiplier=cfg.neighborhood.r_min_multiplier
    )
    r_max, _ = find_r_max(
        dataset,
        r_max_init=cfg.neighborhood.r_max_init,
        num_sweep=cfg.neighborhood.r_max_num_sweep,
        r_min=r_min,
        target_neighborhood_rel_size=cfg.neighborhood.target_neighborhood_rel_size,
    )
    loader = NeighborhoodSegmentLoader(
        dataset,
        cfg.data.segment_length,
        r_min,
        r_max,
        cfg.neighborhood.num_neighbors,
        cfg.neighborhood.num_neighbor_threshold,
        hydra.utils.instantiate(
            cfg.data.batch_strategy,
            batch_size=cfg.data.batch_strategy.batch_size
            // (cfg.neighborhood.num_neighbors + 1),
        ),
    )

    trainer: BaseTrainer = hydra.utils.instantiate(cfg.training)
    trainer.savedir = (
        trainer.savedir
        / f"neighborhood/weight={cfg.neighborhood.weight}"
        / f"neighbors={cfg.neighborhood.num_neighbors}"
        / f"order={cfg.neighborhood.order}"
    )
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    multiterm = True if isinstance(trainer, MultitermTrainer) else False

    loss_fn = NeighborhoodMSELoss2(
        cfg.neighborhood.rollout + 1,
        cfg.neighborhood.weight,
        cfg.neighborhood.chunk_size,
        multiterm=multiterm,
        taylor_order=cfg.neighborhood.order,
        use_taylor_mode=cfg.neighborhood.use_taylor_mode,
        neighbor_position_loss=cfg.neighborhood.neighbor_position_loss,
    )
    model, _ = trainer.train(model, loader, loss_fn, config=config_dict)


if __name__ == "__main__":
    main()
