# halvesting_contrastive/utils/configs/train_config.py

from dataclasses import dataclass
from typing import Literal, Optional

from halvesting_contrastive.utils.helpers import DictAccessMixin


@dataclass
class TrainConfig(DictAccessMixin):
    tau: Optional[float] = None  # Only for CE losses
    # --- optimizer ---
    lr: float = 1e-5
    weight_decay: float = 0.01
    # --- checkpointing ---
    every_n_train_steps: int = 1000
    # --- trainer ---
    gather: bool = True
    device: Literal["cpu", "gpu"] = "gpu"
    num_devices: int = 1
    strategy: str = "ddp_find_unused_parameters_true"
    process_group_backend: Literal["nccl", "gloo", "mpi"] = "gloo"
    max_steps: int = -1
    max_epochs: int = 1
    val_check_interval: Optional[float] = None
    check_val_every_n_epoch: Optional[int] = None
    log_every_n_steps: int = 10
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = None
    precision: Literal["32", "16-mixed"] = "16-mixed"
    overfit_batches: float = 0.0
    # --- wandb ---
    use_wandb: bool = True
    log_model: bool = True
