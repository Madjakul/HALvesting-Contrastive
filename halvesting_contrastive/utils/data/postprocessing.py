# halvesting_geometric/utils/data/postprocessing.py

import logging
import os
from collections import Counter
from typing import List

import datasets
import numpy as np
import torch
from nltk.util import ngrams
from transformers import AutoTokenizer, PreTrainedTokenizer


class Postprocessing:
    """Postprocessing class for the data."""

    device: str
    tokenizer: PreTrainedTokenizer
    max_len: int

    @classmethod
    def set_device(cls, device: str):
        """Set the device for the postprocessing."""
        assert device in ("cpu", "cuda")
        cls.device = device
        return True

    @classmethod
    def set_tokenizer(cls, tokenizer_checkpoint: str, max_len: int):
        """Set the tokenizer for the postprocessing."""
        cls.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        cls.max_len = max_len
        return True

    @classmethod
    def tokenize(cls, batch: List[str]):
        """Tokenize."""
        try:
            return cls.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=cls.max_len,
                return_tensors="pt",
            ).to(cls.device)
        except AttributeError:
            logging.error("Please set the tokenizer and the device first.")
            return None
