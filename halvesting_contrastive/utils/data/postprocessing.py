# halvesting_geometric/utils/data/postprocessing.py

import logging
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import torch
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from transformers import AutoTokenizer, PreTrainedTokenizer

from halvesting_contrastive.modules import Decoder


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
    def compute_unique_ngrams(cls, sentences: List[str], n: int = 3):
        """Compute unique n-gram ratio."""

        def repetition_ratio(sentence: str):
            words = word_tokenize(sentence)
            if len(words) < n:
                return 0.0
            ngram_list = list(ngrams(words, n))
            counts = Counter(ngram_list)
            repeated_ngrams = sum(1 for count in counts.values() if count > 1)
            return repeated_ngrams / len(ngram_list) if len(ngram_list) > 0 else 0.0

        return np.array([repetition_ratio(sentence) for sentence in sentences])

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

    @classmethod
    def postprocess_batch(cls, batch: Dict[str, List[Any]]):
        """Postprocess the batch."""
        query_repetitions = cls.compute_unique_ngrams(batch["query_text"])
        key_repetitions = cls.compute_unique_ngrams(batch["key_text"])

        mask = (
            (np.array([len(query.split()) for query in batch["query_text"]]) > 2)
            & (np.array([len(key.split()) for key in batch["key_text"]]) > 2)
            & (query_repetitions < 0.5)
            & (key_repetitions < 0.5)
        )

        return {k: [v[i] for i in range(len(v)) if mask[i]] for k, v in batch.items()}

    @classmethod
    def entropy_filter_batch(cls, batch: Dict[str, List[Any]], decoder: Decoder):
        """Filter the batch based on entropy."""
        query_tokens = cls.tokenize(batch["query_text"])
        key_tokens = cls.tokenize(batch["key_text"])

        query_entropies = decoder.compute_entropy(query_tokens)
        key_entropies = decoder.compute_entropy(key_tokens)

        mask = (
            (torch.sum(query_tokens["attention_mask"], dim=1) > 4)
            & (torch.sum(key_tokens["attention_mask"], dim=1) > 4)
            & (query_entropies < 0.9)
            & (key_entropies < 0.9)
        )

        return {k: [v[i] for i in range(len(v)) if mask[i]] for k, v in batch.items()}
