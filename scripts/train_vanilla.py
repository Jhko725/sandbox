import hydra
import jax
from dynamics_discovery.data.dataset import TimeSeriesDataset
from dynamics_discovery.data.loaders import SegmentLoader
from dynamics_discovery.loss_functions import MSELoss
from dynamics_discovery.training.vanilla import VanillaTrainer
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", cfg.enable_x64)

    model = hydra.utils.instantiate(cfg.model)
    dataset, _ = (
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
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    model, _ = trainer.train(model, loader, MSELoss(), config=config_dict)

if __name__ == "__main__":
    main()
