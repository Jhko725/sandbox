from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from dynamical_systems.metrics import sinkhorn
from dynamics_discovery.data.dataset import TimeSeriesDataset
from dynamics_discovery.misc import load_experiment, solve_batch
from omegaconf import DictConfig


@hydra.main(
    config_path="./configs_analysis", config_name="config_sinkhorn", version_base=None
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
    idx_long = np.asarray([1, 10, 100, 1000, 10000]) - 1
    t_norm = np.arange(dataset_test.t.shape[0]) * 0.01
    t_long, t_long_norm = dataset_test.t[0, idx_long], t_norm[idx_long]
    u_long_batch = dataset_test.u[:, idx_long]
    del dataset_test

    model = load_experiment(
        cfg.experiment,
        cfg.key,
        cfg.downsample_factor,
        cfg.train_length,
        cfg.noise_std_relative,
        cfg.data.checkpointpath,
    )

    u_long_pred = solve_batch(model, t_long, u_long_batch[:, 0])
    metric = []
    for i in range(1, len(idx_long)):
        metric.append(sinkhorn(u_long_batch[:, i], u_long_pred[:, i]))

    savedir = Path(cfg.savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    jnp.save(savedir / cfg.savename, np.asarray(metric))


if __name__ == "__main__":
    main()
