import logging
import warnings
from typing import List, Sequence, Mapping
from pathlib import Path

import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

import torch
import re


def is_list(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


def is_dict(x):
    return isinstance(x, Mapping)


def to_dict(x, recursive=True):
    """Convert Sequence or Mapping object to dict
    lists get converted to {0: x[0], 1: x[1], ...}
    """
    if is_list(x):
        x = {i: v for i, v in enumerate(x)}
    if is_dict(x):
        if recursive:
            return {k: to_dict(v, recursive=recursive) for k, v in x.items()}
        else:
            return dict(x)
    else:
        return x


def to_list(x, recursive=False):
    """Convert an object to list.
    If Sequence (e.g. list, tuple, Listconfig): just return it
    Special case: If non-recursive and not a list, wrap in list
    """
    if is_list(x):
        if recursive:
            return [to_list(_x) for _x in x]
        else:
            return list(x)
    else:
        if recursive:
            return x
        else:
            return [x]


def last_modification_time(path):
    """Including files / directory 1-level below the path
    """
    path = Path(path)
    if path.is_file():
        return path.stat().st_mtime
    elif path.is_dir():
        return max(child.stat().st_mtime for child in path.iterdir())
    else:
        return None


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if config.get("experiment_mode") and not config.get("name"):
        log.info(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py mode=exp name=experiment_name`"
        )
        log.info("Exiting...")
        exit()

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.trainer.get("fast_dev_run"):
        log.info(
            "Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>"
        )
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "train",
        "eval",
        "callbacks",
        "logger",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def load_checkpoint(path, device="cpu"):
    path = Path(path).expanduser()
    is_deepspeed = False
    if path.is_dir():  # DeepSpeed checkpoint
        is_deepspeed = True
        latest_path = path / "latest"
        if latest_path.is_file():
            with open(latest_path, "r") as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")
        path /= f"{tag}/mp_rank_00_model_states.pt"
    state_dict = torch.load(path, map_location=device)
    if is_deepspeed:
        state_dict = state_dict["module"]

        # Replace the names of some of the submodules
        def key_mapping(key):
            return re.sub(r"^module.model.", "", key)

        state_dict = {key_mapping(k): v for k, v in state_dict.items()}
    return state_dict
