# halvesting_contrastive/utils/train_utils.py

import os.path as osp
from typing import Optional

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from halvesting_contrastive.utils.configs.base_config import BaseConfig
from halvesting_contrastive.utils.data.halvest_contrastive_datamodule import (
    HALvestContrastiveDatamodule,
)


def setup_datamodule(
    cfg: BaseConfig,
    processed_ds_dir: str,
    num_proc: int,
    cache_dir: Optional[str] = None,
) -> L.LightningDataModule:
    dm = HALvestContrastiveDatamodule(
        cfg=cfg,
        processed_ds_dir=processed_ds_dir,
        num_proc=num_proc,
        cache_dir=cache_dir,
    )
    return dm


def setup_trainer(
    cfg: BaseConfig,
    logs_dir: str,
    checkpoint_dir: Optional[str] = None,
) -> L.Trainer:
    # Set up callbacks
    callbacks = []

    name = (
        f"{cfg.model.base_model_name}-{cfg.data.subsets[0]}-gather:{cfg.train.gather}"
    ).replace("/", "_")

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Model checkpoint callback if checkpoint_dir is provided
    if checkpoint_dir is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=osp.join(checkpoint_dir, name),
            filename="{step}",
            every_n_train_steps=cfg.train.every_n_train_steps,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    # Configure loggers
    loggers = []
    if cfg.train.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            name=name,
            log_model=cfg.train.log_model,
            group=cfg.group_name,
            config=cfg.to_dict(),
        )
        loggers.append(wandb_logger)

    # Add CSV logger by default
    csv_logger = CSVLogger(save_dir=logs_dir, name=name)
    loggers.append(csv_logger)

    if cfg.train.strategy.startswith("ddp"):
        strategy = DDPStrategy(
            find_unused_parameters=cfg.train.strategy.endswith(
                "find_unused_parameters_true"
            ),
            process_group_backend=cfg.train.process_group_backend,
        )
    else:
        strategy = cfg.train.strategy

    trainer = L.Trainer(
        accelerator=cfg.train.device,
        strategy=strategy,
        num_sanity_val_steps=0,
        devices=cfg.train.num_devices,
        max_steps=cfg.train.max_steps,
        max_epochs=cfg.train.max_epochs,
        val_check_interval=0.02,
        # check_val_every_n_epoch=None,
        enable_checkpointing=checkpoint_dir is not None,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=cfg.train.log_every_n_steps,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        gradient_clip_val=cfg.train.gradient_clip_val,
        precision=cfg.train.precision,
        overfit_batches=cfg.train.overfit_batches,
    )
    return trainer
