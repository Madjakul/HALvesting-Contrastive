# halvesting_contrastive/core/ict_sampler.py

import logging
import random
from typing import Any, Dict, List

from nltk.tokenize import sent_tokenize


class ICTSampler:
    """Class used to sample documents and grammatically correct passages from
    the same document.

    The sampling is done in a batched manner to speed up the process.
    The sampling is done in the following way: for each document, sample `n`
    contiguous sentences as the query and `n` contiguous sentences before and
    after the query as the key. The labels are always 1.
    """

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

        for idx in range(len(batch["text"])):  # type: ignore
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
        """Sample `n` sentences from a text. A query consists in `n` contiguous
        sentences and the key consists in the `n` contiguous sentences before
        and after the query.

        Parameters
        ----------
        text: str
            Text from which to sample sentences.
        n_sentences: int
            Number of sentences to sample for the query. This number is multiplied
            by 2 to get they keys on the left and right of the query.

        Returns
        -------
        query: str
            Query text.
        key: str
            Key text.
        """
        sentences = sent_tokenize(text)
        query_idx = random.randint(n_sentences, len(sentences) - 2 * n_sentences)
        query = " ".join(sentences[query_idx : query_idx + n_sentences])
        key = " ".join(
            sentences[query_idx - n_sentences : query_idx]
            + sentences[query_idx + n_sentences : query_idx + 2 * n_sentences]
        )
        return query, key
