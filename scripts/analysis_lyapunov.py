from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from dynamics_discovery.data.dataset import TimeSeriesDataset
from dynamics_discovery.misc import calculate_lyapunov, load_experiment
from omegaconf import DictConfig


@hydra.main(
    config_path="./configs_analysis", config_name="config_lyapunov", version_base=None
)
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", True)

    _, transform = (
        TimeSeriesDataset.from_hdf5(cfg.data.trainpath)
        .downsample(cfg.downsample_factor)
        .add_noise(cfg.noise_std_relative)
        .standardize()
    )

    dataset_test = (
        TimeSeriesDataset.from_hdf5(cfg.data.testpath)
        .add_noise(cfg.noise_std_relative)
        .apply_transform(transform)
    )
    u0 = dataset_test.u[:, 0]
    del dataset_test

    lya_true_dict = jnp.load(cfg.data.testpath_lyapunov)
    t = lya_true_dict["t"][::1]

    model = load_experiment(
        cfg.experiment,
        cfg.key,
        cfg.downsample_factor,
        cfg.train_length,
        cfg.noise_std_relative,
        cfg.data.checkpointpath,
    )
    lyapunovs = calculate_lyapunov(model, t, u0, batch_size=5000)

    savedir = Path(cfg.savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    jnp.save(savedir / cfg.savename, np.asarray(lyapunovs))


if __name__ == "__main__":
    main()
