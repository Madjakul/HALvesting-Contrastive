# halvesting_contrastive/modules/decoder.py

import logging
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class Decoder:
    """Decoder class for the model."""

    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def compute_entropy(self, tokenized_inputs: Dict[str, Any]):
        """Compute the average entropy of a sequence."""
        logits = self.model(**tokenized_inputs).logits
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) / torch.log(
            torch.tensor(self.model.config.vocab_size).to(self.device)
        )

        attention_mask = tokenized_inputs["attention_mask"]
        seq_len = torch.sum(attention_mask, dim=1)
        mean_entropy = torch.sum(entropy * attention_mask, dim=1) / seq_len
        return mean_entropy.cpu().numpy()
