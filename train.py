import shutil
import typing as tp
import logging
from logging import Formatter
import sys
import traceback

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler

from data import musdbDataset, collate_fn
# from model import BandSplitRNN, PLModel

log = logging.getLogger(__name__)

def initialize_loaders(cfg: DictConfig) -> tp.Tuple[DataLoader, DataLoader]:
    """
    Initializes train and validation dataloaders from configuration file.
    """
    train_dataset = musdbDataset(
        **cfg.train_dataset,
    )
    train_loader = DataLoader(
        train_dataset,
        **cfg.train_loader,
        collate_fn=collate_fn
    )
    if hasattr(cfg, 'val_dataset'):
        val_dataset = musdbDataset(
            **cfg.val_dataset,
        )
        val_loader = DataLoader(
            val_dataset,
            **cfg.val_loader,
            collate_fn=collate_fn
        )
    else:
        val_loader = None
    return (
        train_loader,
        val_loader
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    pl.seed_everything(42, workers=True)

    print(OmegaConf.to_yaml(cfg))

    # log.info()

    log.info("Initializing loaders, featurizers.")
    train_loader, val_loader = initialize_loaders(cfg)

    # featurizer, inverse_featurizer = initialize_featurizer(cfg)
    # augs = initialize_augmentations(cfg)
    #
    # log.info("Initializing model, optimizer, scheduler.")
    # model, opt, sch = initialize_model(cfg)
    #
    # log.info("Initializing Lightning logger and callbacks.")
    # logger, callbacks = initialize_utils(cfg)
    #
    # log.info("Initializing Lightning modules.")
    # plmodel = PLModel(
    #     model,
    #     featurizer, inverse_featurizer,
    #     augs,
    #     opt, sch,
    #     cfg
    # )
    # trainer = pl.Trainer(
    #     **cfg.trainer,
    #     logger=logger,
    #     callbacks=callbacks,
    # )
    #
    # log.info("Starting training...")
    # try:
    #     trainer.fit(
    #         plmodel,
    #         train_dataloaders=train_loader,
    #         val_dataloaders=val_loader,
    #         ckpt_path=cfg.ckpt_path
    #     )
    # except Exception as e:
    #     log.error(traceback.format_exc())
    #
    # log.info("Training finished!")
    #
    # if cfg.trainer.fast_dev_run:
    #     hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    #     shutil.rmtree(hydra_cfg['runtime']['output_dir'])


if __name__ == "__main__":
    my_app()