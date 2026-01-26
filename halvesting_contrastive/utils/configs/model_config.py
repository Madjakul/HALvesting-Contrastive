# halvesting_contrastive/utils/configs/model_config.py

from dataclasses import dataclass

from halvesting_contrastive.utils.helpers import DictAccessMixin


@dataclass
class ModelConfig(DictAccessMixin):
    base_model_name: str = "FacebookAI/roberta-base"
