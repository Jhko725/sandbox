import mmap
from collections.abc import Mapping
from pathlib import Path

import equinox as eqx
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .custom_types import PathLike


def save_model(
    filepath: PathLike,
    model: eqx.Module,
    config: Mapping | None,
    overwrite: bool = False,
):
    """
    Save an `equinox.Module` and the associated configuration data
    using `equinox` and `omegaconf`.

    Inspired from https://docs.kidger.site/equinox/api/serialisation/
    """
    filepath = Path(filepath)

    config = OmegaConf.create(config)

    if filepath.exists() and overwrite:
        filepath.unlink()

    with open(filepath, "x") as file:
        OmegaConf.save(config, file, resolve=True)
        file.write("\n#!\n")
    with open(filepath, "ab") as file:
        eqx.tree_serialise_leaves(file, model)


def load_model(filepath: PathLike) -> eqx.Module:
    with open(filepath, "r+") as file:
        with mmap.mmap(file.fileno(), 0) as file_mmap:
            delimiter = b"\n#!\n"
            split_ind = file_mmap.find(delimiter)
            config = OmegaConf.create(file_mmap[:split_ind].decode())
            model = instantiate(config)

    with open(filepath, "rb") as file:
        _ = file.read(split_ind + len(delimiter))
        model = eqx.tree_deserialise_leaves(file, model)
    return model
