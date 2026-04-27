import hydra
import jax
from dynamics_discovery.data.dataset import TimeSeriesDataset
from dynamics_discovery.loss_functions import AbstractDynamicsLoss
from dynamics_discovery.training.vanilla import VanillaTrainer
from omegaconf import DictConfig, OmegaConf


jax.config.update("jax_enable_x64", True)


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Manually select gpu to run on
    if cfg.gpu_id != "auto":
        jax.config.update("jax_default_device", jax.devices("gpu")[cfg.gpu_id])

    model = hydra.utils.instantiate(cfg.model)
    # time_slice = slice(0, 500 * cfg.data.downsample_factor, cfg.data.downsample_factor)

    dataset_train, transform = TimeSeriesDataset.from_hdf5(
        cfg.data.dataset.loadpath.train,  # time_slice=time_slice
    ).standardize()
    dataset_train = dataset_train.add_noise(cfg.data.noise_std_relative)

    dataset_valid = TimeSeriesDataset.from_hdf5(
        cfg.data.dataset.loadpath.validation,  # time_slice=time_slice
    ).apply_transform(transform)

    loader_partial = hydra.utils.instantiate(cfg.method.loader_partial)
    loader_train = loader_partial(dataset_train)
    loader_valid = loader_partial(dataset_valid)

    trainer: VanillaTrainer = hydra.utils.instantiate(cfg.training)
    trainer.savedir = trainer.savedir / cfg.method.subsavedir
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    loss_fn: AbstractDynamicsLoss = hydra.utils.instantiate(cfg.method.loss_fn)

    model, _ = trainer.train(
        model, loss_fn, loader_train, loader_valid, config=config_dict
    )


if __name__ == "__main__":
    main()
