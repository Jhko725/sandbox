import mmap
from collections.abc import Mapping
from pathlib import Path

import equinox as eqx
import more_itertools
import orbax.checkpoint as ocp
from hydra.utils import instantiate
from jaxtyping import PyTree
from omegaconf import DictConfig, OmegaConf

from .custom_types import PathLike


def save_model(
    model: PyTree,
    config: dict | OmegaConf,
    savedir: str | Path,
    step_number: int = 0,
    options: ocp.CheckpointManagerOptions | None = None,
) -> None:
    """
    Saves a given PyTree using `orbax.checkpoint` and `hydra`.

    It is assumed that the model PyTree can be constructed by calling
    `hydra.utils.instantiate` on the provided config dictionary/DictConfig.
    This information is saved in a json format, as a top level metadata of the
    checkpoint.

    The model weights are saved using `orbax.checkpoint`'s built-in support for
    PyTrees of Arrays.

    While this function is blocking, one can utilize the logics contained in this
    function to perform nonblocking saves (as done in `../training.base.py`).
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if options is None:
        options = ocp.CheckpointManagerOptions()

    with ocp.CheckpointManager(
        Path(savedir).resolve(), options=options, metadata=config
    ) as mngr:
        mngr.save(
            step_number, args=ocp.args.StandardSave(eqx.filter(model, eqx.is_array))
        )


def _infer_step(ckpt_dir: Path) -> int:
    """Given a checkpoint directory created by orbax.checkpoint consisting of a
    checkpoint from a single train step, determine and return the corresponding step
    number."""
    ckpt_step_dirs = filter(
        lambda p: p.is_dir() and p.stem.isnumeric(), ckpt_dir.iterdir()
    )
    try:
        return int(more_itertools.one(ckpt_step_dirs).stem)
    except ValueError:
        raise ValueError(
            """Given directory contains checkpoints from multiple steps, and thus step 
            cannot be inferred."""
        )


def load_model(
    loaddir: str | Path,
    step_number: int | None = None,
    options: ocp.CheckpointManagerOptions | None = None,
) -> PyTree:
    """Loads model from a checkpoint created by the `save_model` function.

    loaddir: Directory containing the checkpoints
    step_number: Number corresponding to the specific checkpoint
    """
    options = ocp.CheckpointManagerOptions() if options is None else options

    loaddir = Path(loaddir).resolve()
    if step_number is None:
        step_number = _infer_step(loaddir)

    with ocp.CheckpointManager(loaddir, options=options) as mngr_load:
        model_config = mngr_load.metadata().custom_metadata
        model_backbone = instantiate(OmegaConf.create(model_config))

        weights_backbone, rest = eqx.partition(model_backbone, eqx.is_array_like)
        weights_load = mngr_load.restore(
            step_number,
            args=ocp.args.StandardRestore(weights_backbone),
        )
    return eqx.combine(weights_load, rest)


## Save/load functions using equinox+hydra. Good for simple cases.


def save_model_eqx(
    filepath: PathLike,
    model: eqx.Module,
    config: Mapping | None,
    overwrite: bool = False,
):
    """
    Save an `equinox.Module` and the associated configuration data
    using `equinox` and `omegaconf`.

    Inspired from https://docs.kidger.site/equinox/examples/serialisation/
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


def load_model_eqx(filepath: PathLike) -> eqx.Module:
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
