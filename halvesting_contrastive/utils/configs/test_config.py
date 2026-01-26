# halvesting_contrastive/utils/configs/test_config.py

from dataclasses import dataclass
from typing import Literal

from halvesting_contrastive.utils.helpers import DictAccessMixin


@dataclass
class TestConfig(DictAccessMixin):
    subset: str = "ict-1"
    # --- trainer ---
    device: Literal["cpu", "gpu"] = "gpu"
    num_devices: int = 1
    process_group_backend: Literal["nccl", "gloo", "mpi"] = "gloo"
    # --- wandb ---
    use_wandb: bool = True
