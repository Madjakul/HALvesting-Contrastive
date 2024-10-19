# halvesting_contrastive/core/sampler.py

import logging
from typing import Optional

import datasets
import torch


class Sampler:
    """Class used to sample documents and grammatically correct passages from
    documents."""

    def __init__(self, dataset: datasets.Dataset):
        self.dataset = dataset

    def _compute_probs(self):
        probs = None

    def sample_document(self):
        raise NotImplementedError
