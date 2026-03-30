from functools import partial
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import xarray as xr
from dynamical_systems.analysis import lyapunov_gr
from dynamical_systems.continuous import solve_ode
from hydra.core.hydra_config import HydraConfig
from jaxtyping import Array, Float
from omegaconf import DictConfig


jax.config.update("jax_enable_x64", True)


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    system_name = hydra_cfg.runtime.choices["system"]
    system_name = system_name.rstrip(".yaml")
    if cfg.gpu_id != "auto":
        jax.config.update("jax_default_device", jax.devices("gpu")[cfg.gpu_id])

    ode = hydra.utils.instantiate(cfg.system.ode)

    @partial(eqx.filter_vmap, in_axes=(None, 0))
    def solve_ode_batch(
        _ts: Float[Array, " T"], _u0: Float[Array, "B D"]
    ) -> Float[Array, "B T D"]:
        return solve_ode(
            ode, _ts, _u0, rtol=cfg.system.rtol, atol=cfg.system.atol, max_steps=None
        )

    def get_on_attractor_points(
        _key, num_points: int
    ) -> Float[Array, "{num_points} {ode.dim}"]:
        # Sample off-attractor initial points from an isotropic gaussian
        u0_ = jax.random.normal(_key, shape=(num_points, ode.dim))
        u0_ = u0_ * cfg.system.init_std + cfg.system.init_mean
        return solve_ode_batch(jnp.asarray([0, cfg.system.burnin_time]), u0_)[:, -1]

    def estimate_lyapunov_spectrum(_key, num_trajs: int) -> Float[Array, "{ode.dim}"]:
        u0 = get_on_attractor_points(_key, num_trajs)
        ts = jnp.arange(0, cfg.system.t_max_lyapunov, cfg.system.dt_lyapunov)

        lyas: Float[Array, "trajs ts dim"] = eqx.filter_vmap(
            lambda _u0: lyapunov_gr(
                ode, _u0, ts, rtol=cfg.system.rtol, atol=cfg.system.atol
            )[0]
        )(u0)
        return jnp.mean(lyas[:, -1], axis=0)

    def generate_trajectories(
        _key, num_trajs: int, ts: Float[Array, " time"]
    ) -> Float[Array, "{num_trajs} time {ode.dim}"]:
        u0 = get_on_attractor_points(_key, num_trajs)
        return solve_ode_batch(ts, u0)

    def to_xarray(
        ts: Float[Array, " time"],
        us: Float[Array, "traj time dim"],
        lyapunovs: Float[Array, " dim"],
    ) -> xr.Dataset:
        trajectories = xr.DataArray(
            data=us, dims=["traj", "time", "dim"], coords={"time": ts}
        )
        dataset = xr.Dataset(
            data_vars={"trajectories": trajectories, "lyapunovs": (["dim"], lyapunovs)}
        )
        return dataset

    key = jax.random.key(cfg.key)

    for i, (dataset_type, num_trajs) in enumerate(cfg.num_trajectories.items()):
        subkey = jax.random.fold_in(key, i)
        key_lya, key_data = jax.random.split(subkey)

        # Estimate Lyapunov spectrum
        lyapunovs = estimate_lyapunov_spectrum(key_lya, num_trajs)

        lya_max = jnp.max(lyapunovs)
        ts = jnp.arange(0, cfg.t_max_norm / lya_max, cfg.dt_norm / lya_max)
        us = generate_trajectories(key_data, num_trajs, ts)

        dataset: xr.Dataset = to_xarray(ts, us, lyapunovs)

        savedir = Path(cfg.savedir)
        filename = f"{system_name}_{dataset_type}.hdf5"

        dataset.to_netcdf(savedir / filename, engine="h5netcdf")


if __name__ == "__main__":
    main()
