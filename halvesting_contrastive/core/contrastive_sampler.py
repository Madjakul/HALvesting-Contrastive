# halvesting_contrastive/core/contra_sampler.py

import logging
import os
import random
from collections import defaultdict
from typing import Dict, List

import datasets
import torch
from nltk.tokenize import sent_tokenize

from halvesting_contrastive.core.formater import Formater


class ContrastiveSampler:
    """Class used to sample documents and passages for contrastive learning."""

    _auth_to_idx: Dict[str, List[int]] = {}

    @classmethod
    def _build_auth_to_idx_map(cls, ds: datasets.Dataset):
        """Build a dictionary: author -> [list of indices] where that author appears in the dataset."""
        auth_to_idx = defaultdict(list)
        for idx, auths in enumerate(ds["authorids"]):
            for auth in auths:
                auth_to_idx[auth].append(idx)
        return auth_to_idx

    @classmethod
    def init_cache(cls, ds: datasets.Dataset):
        """Initialize the cached author-to-indices map once."""
        if cls._auth_to_idx is None:
            cls._auth_to_idx = cls._build_auth_to_idx_map(ds)
        return cls._auth_to_idx

    @classmethod
    def sample_batched(
        cls,
        batch: List[datasets.Dataset],
        n_pairs: int,
        n_sentences: int,
        ds: datasets.Dataset,
        all_ids: List[int],
    ):
        """Batched function to create n positive pairs and n negative pairs per
        documents."""
        query_halids = []
        query_texts = []
        query_years = []
        query_domains = []
        query_affiliations = []
        query_authors = []
        passage_halids = []
        passage_texts = []
        passage_years = []
        passage_domains = []
        passage_affiliations = []
        passage_authors = []
        domain_labels = []
        affiliation_labels = []
        author_labels = []

        for idx, doc in batch:
            pass
