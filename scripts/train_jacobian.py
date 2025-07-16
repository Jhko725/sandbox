import hydra
import jax
from dynamical_systems.transforms import TransformedODE
from dynamics_discovery.data import AllSegmentLoader, RandomSegmentLoader
from dynamics_discovery.data.dataset import TimeSeriesDataset
from dynamics_discovery.loss_functions import JacobianMatchingMSE
from dynamics_discovery.training.vanilla import VanillaTrainer
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./configs", config_name="config_jacobian", version_base=None)
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", cfg.enable_x64)

    model = hydra.utils.instantiate(cfg.model)

    dataset, transform = (
        TimeSeriesDataset.load(cfg.data.dataset.loadpath)
        .downsample(cfg.data.downsample_factor)
        .add_noise(cfg.data.noise_std_relative)
        .standardize()
    )

    if cfg.data.load_strategy == "random":
        loader = RandomSegmentLoader(
            dataset, cfg.data.segment_length, cfg.data.batch_size
        )
    elif cfg.data.load_strategy == "full":
        loader = AllSegmentLoader(dataset, cfg.data_segment_length)

    trainer: VanillaTrainer = hydra.utils.instantiate(cfg.training)
    trainer.savedir = trainer.savedir / "jacobian/"
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    ode_true = hydra.utils.instantiate(cfg.data.dataset.ode)
    loss_fn = JacobianMatchingMSE(
        TransformedODE(ode_true, transform), cfg.jacobian.weight
    )
    model, _ = trainer.train(model, loader, loss_fn, config=config_dict)
    trainer.save_model(model, config_dict["model"])


if __name__ == "__main__":
    main()
