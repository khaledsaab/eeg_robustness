from typing import Callable
import dotenv
from typing import List, Sequence
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import get_logger, last_modification_time, extras, print_config

log = get_logger(__name__)

dotenv.load_dotenv(override=True)

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
# Delay the evaluation until we have the datamodule
# So we want the resolver to yield the same string.
OmegaConf.register_new_resolver(
    "datamodule", lambda attr: "${datamodule:" + str(attr) + "}"
)


def train(config: DictConfig):
    """

    Outline training code adapted from https://github.com/HazyResearch/zoo

    Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    """
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # We want to add fields to config so need to call OmegaConf.set_struct
    OmegaConf.set_struct(config, False)

    # Init lightning model
    model: LightningModule = hydra.utils.instantiate(
        config.task, cfg=config, _recursive_=False
    )
    datamodule: LightningDataModule = model._datamodule

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if cb_conf is not None and "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if lg_conf is not None and "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    ckpt_cfg = {}
    if config.get("resume"):
        try:
            checkpoint_path = Path(config.callbacks.model_checkpoint.dirpath)
            if checkpoint_path.is_dir():
                last_ckpt = checkpoint_path / "last.ckpt"
                autosave_ckpt = checkpoint_path / ".pl_auto_save.ckpt"
                if not (last_ckpt.exists() or autosave_ckpt.exists()):
                    raise FileNotFoundError(
                        "Resume requires either last.ckpt or .pl_autosave.ckpt"
                    )
                if (not last_ckpt.exists()) or (
                    autosave_ckpt.exists()
                    and last_modification_time(autosave_ckpt)
                    > last_modification_time(last_ckpt)
                ):
                    # autosave_ckpt = autosave_ckpt.replace(autosave_ckpt.with_name('.pl_auto_save_loaded.ckpt'))
                    checkpoint_path = autosave_ckpt
                else:
                    checkpoint_path = last_ckpt
            # DeepSpeed's checkpoint is a directory, not a file
            if checkpoint_path.is_file() or checkpoint_path.is_dir():
                ckpt_cfg = {"ckpt_path": str(checkpoint_path)}
            else:
                log.info(
                    f"Checkpoint file {str(checkpoint_path)} not found. Will start training from scratch"
                )
        except (KeyError, FileNotFoundError):
            pass

    # Configure ddp automatically
    n_devices = config.trainer.get("devices", 1)
    if isinstance(n_devices, Sequence):  # trainer.devices could be [1, 3] for example
        n_devices = len(n_devices)
    if n_devices > 1 and config.trainer.get("strategy", None) is None:
        config.trainer.strategy = dict(
            _target_="pytorch_lightning.strategies.DDPStrategy",
            find_unused_parameters=True,
            gradient_as_bucket_view=True,  # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
        )

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger
    )

    # Train the model
    log.info("Starting training!")

    trainer.fit(model=model, datamodule=datamodule, **ckpt_cfg)
    #trainer.fit(model=model, **ckpt_cfg)

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule)


def dictconfig_filter_key(d: DictConfig, fn: Callable) -> DictConfig:
    """Only keep keys where fn(key) is True. Support nested DictConfig."""
    # Using d.items_ex(resolve=False) instead of d.items() since we want to keep the
    # ${datamodule:foo} unresolved for now.
    return DictConfig(
        {
            k: dictconfig_filter_key(v, fn) if isinstance(v, DictConfig) else v
            # for k, v in d.items_ex(resolve=False) if fn(k)})
            for k, v in d.items()
            if fn(k)
        }
    )


@hydra.main(config_path="configs/", config_name="config.yaml")
def run(config: DictConfig):
    # Remove config keys that start with '__'. These are meant to be used only in computing
    # other entries in the config.
    config = dictconfig_filter_key(config, lambda k: not k.startswith("__"))

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    run()
