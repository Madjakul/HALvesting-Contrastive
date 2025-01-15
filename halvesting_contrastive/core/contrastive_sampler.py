# halvesting_contrastive/core/contra_sampler.py

import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

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
        ids: List[int],
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
            local_idx = ids[idx]
            local_auths = set(doc["authorids"])

            same_auth_ids = set()
            for auth in local_auths:
                same_auth_ids.update(cls._auth_to_idx[auth])

            same_auth_ids.discard(local_idx)
            diff_auth_ids = set(all_ids) - same_auth_ids - {local_idx}

            # Sample positive pairs
            for _ in range(n_pairs):
                query_text = cls.sample_sentences(doc, n_sentences)
                passage_idx = (
                    random.choice(list(same_auth_ids)) if same_auth_ids else local_idx
                )
                passage = ds[passage_idx]
                passage_text = cls.sample_sentences(passage, n_sentences)

    @staticmethod
    def sample_sentences(document: Dict[str, Any], n_sentences: int):
        """Sample n sentences from a document."""
        sentences = sent_tokenize(document["text"])
        sentence_idx = torch.randint(len(sentences), (1,)).item()
        while sentence_idx > len(sentences) - n_sentences:
            sentence_idx = torch.randint(len(sentences), (1,)).item()
        sentence = " ".join(sentences[sentence_idx : sentence_idx + n_sentences])

        return sentence
