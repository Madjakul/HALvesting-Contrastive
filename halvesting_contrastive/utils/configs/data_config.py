# halvesting_contrastive/utils/configs/data_config.py

from dataclasses import dataclass, field
from typing import List

from halvesting_contrastive.utils.helpers import DictAccessMixin


@dataclass
class DataConfig(DictAccessMixin):
    subsets: List[str] = field(default_factory=list)
    batch_size: int = 64
    tokenizer_name: str = "FacebookAI/roberta-base"
    max_length: int = 512
    map_batch_size: int = 1000
    load_from_cache_file: bool = False
    shuffle: bool = True
