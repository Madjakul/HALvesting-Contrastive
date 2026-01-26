# halvesting_contrastive/modules/language_model.py

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from transformers import AutoConfig, AutoModelForMaskedLM

if TYPE_CHECKING:
    from halvesting_contrastive.utils.configs import BaseConfig


class LanguageModel(nn.Module):

    def __init__(self, cfg: "BaseConfig") -> None:
        super(LanguageModel, self).__init__()
        self.cfg = cfg

        config = AutoConfig.from_pretrained(self.cfg.model.base_model_name)
        logging.info(
            f"Loading pretrained encoder from {self.cfg.model.base_model_name}."
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.cfg.model.base_model_name, config=config
        )

        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch seq"],
        attention_mask: Int[torch.Tensor, "batch seq"],
    ) -> Float[torch.Tensor, "batch seq hidden"]:

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states[-1]
