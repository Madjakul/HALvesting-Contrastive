# halvesting_contrastive/core/ict_sampler.py

import logging
import random
from typing import Any, Dict, List

from nltk.tokenize import sent_tokenize


# TODO: implement the class and test it
class ICTSampler:
    """Class used to sample documents and grammatically correct passages
    from."""

    @classmethod
    def sample_batched(
        cls, batch: List[Dict[str, Any]], n_pairs: int, n_sentences: int = 1
    ):
        query_halids = []
        query_texts = []
        query_years = []
        query_domains = []
        query_affiliations = []
        query_authors = []
        key_texts = []
        domain_labels = []
        affiliation_labels = []
        author_labels = []

        for idx in range(len(batch)):
            for _ in range(n_pairs):
                query_text, key_text = cls.sample_sentences(batch["text"][idx], n_sentences)  # type: ignore
                query_halids.append(batch["halid"][idx])  # type: ignore
                query_texts.append(query_text)
                query_years.append(batch["year"][idx])  # type: ignore
                query_domains.append(batch["domain"][idx])  # type: ignore
                query_affiliations.append(batch["affiliations"][idx])  # type: ignore
                query_authors.append(batch["authorids"][idx])  # type: ignore
                key_texts.append(key_text)
                domain_labels.append(1)
                affiliation_labels.append(1)
                author_labels.append(1)
        return {
            "query_halid": query_halids,
            "query_text": query_texts,
            "query_year": query_years,
            "query_authors": query_authors,
            "query_affiliations": query_affiliations,
            "query_domains": query_domains,
            "key_halid": query_halids,
            "key_text": key_texts,
            "key_year": query_years,
            "key_authors": query_authors,
            "key_affiliations": query_affiliations,
            "key_domains": query_domains,
            "domain_label": domain_labels,
            "affiliation_label": affiliation_labels,
            "author_label": author_labels,
        }

    @staticmethod
    def sample_sentences(text: str, n_sentences: int):
        """Sample n sentences from a text."""
        sentences = sent_tokenize(text)
        query_idx = random.randint(n_sentences, len(sentences) - 2 * n_sentences)
        query = " ".join(sentences[query_idx : query_idx + n_sentences])
        key = " ".join(
            sentences[query_idx - n_sentences : query_idx]
            + sentences[query_idx + n_sentences : query_idx + 2 * n_sentences]
        )
        return query, key
