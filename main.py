import hydra
import jax
from dynamics_discovery.dataset import TimeSeriesDataset
from dynamics_discovery.preprocessing import (
    add_noise,
    downsample,
    split_into_chunks,
    standardize,
)
from dynamics_discovery.training.vanilla import VanillaTrainer
from dynamics_discovery.utils.tree import tree_satisfy_float_precision
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", cfg.enable_x64)

    model = hydra.utils.instantiate(cfg.model)
    dataset = TimeSeriesDataset.load(cfg.data.loadpath)

    if not tree_satisfy_float_precision(model, dataset, expect_x64=cfg.enable_x64):
        raise TypeError(
            """Model and/or dataset does not conform to the 
            expected floating point precision!"""
        )
    t_train = dataset.t[0][:: cfg.preprocessing.downsample.keep_every]
    u_train = downsample(dataset.u[0], cfg.preprocessing.downsample.keep_every)
    u_train = add_noise(
        u_train,
        cfg.preprocessing.noise.rel_noise_strength,
        cfg.preprocessing.noise.preserve_first,
        cfg.preprocessing.noise.key,
    )
    u_train = standardize(u_train)

    t_train_batched, u_train_batched = jax.tree.map(
        lambda x: split_into_chunks(
            x, cfg.preprocessing.batch_length, cfg.preprocessing.overlap
        ),
        (t_train, u_train),
    )

    trainer: VanillaTrainer = hydra.utils.instantiate(cfg.training)
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    model, _ = trainer.train(
        model, t_train_batched, u_train_batched, config=config_dict
    )
    trainer.save_model(model, config_dict["model"])


if __name__ == "__main__":
    main()
