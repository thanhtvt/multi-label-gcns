from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in @task_wrapper decorator
    which applies extra utilities before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with instantiated objects
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule: <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model: <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    loggers: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.logger)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": loggers,
        "trainer": trainer,
    }

    if loggers:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting evaluation...")
    trainer.test(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions
    # trainer.predict(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict
