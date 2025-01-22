# halvesting_contrastive/core/contrastive_sampler.py

import logging
import random
from collections import defaultdict
from typing import Any, Dict, List

import datasets
import torch
from nltk.tokenize import sent_tokenize


# TODO: Document the class and the functions.
class ContrastiveSampler:
    """Class used to sample documents and keys for contrastive learning."""

    _auth_to_idx: Dict[str, List[int]] = None  # type: ignore

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
            logging.info("Building author-to-indices map...")
            cls._auth_to_idx = cls._build_auth_to_idx_map(ds)
        logging.info("Author-to-indices map built.")
        return True

    @classmethod
    def sample_batched(
        cls,
        batch: List[Dict[str, Any]],
        ids: List[int],
        soft_positives: bool,
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
        key_halids = []
        key_texts = []
        key_years = []
        key_domains = []
        key_affiliations = []
        key_authors = []
        domain_labels = []
        affiliation_labels = []
        author_labels = []

        for idx in range(len(batch["text"])):
            local_idx = ids[idx]

            same_auth_ids = set()
            if soft_positives:
                local_auths = set(batch["authorids"][idx])  # type: ignore
                for auth in local_auths:
                    same_auth_ids.update(cls._auth_to_idx[auth])
                diff_auth_ids = set(all_ids) - same_auth_ids - {local_idx}
            else:
                local_auths = batch["authorids"][idx]  # type: ignore
                same_auth_ids.update(cls._auth_to_idx[local_auths[0]])
                for idx_ in list(same_auth_ids):
                    if ds[idx_]["authorids"][0] != local_auths[0]:
                        same_auth_ids.discard(idx_)
                common_auths = set()
                for auth in local_auths:
                    common_auths.update(cls._auth_to_idx[auth])
                diff_auth_ids = (
                    set(all_ids) - same_auth_ids - common_auths - {local_idx}
                )

            same_auth_ids.discard(local_idx)

            # Sample positive pairs
            for _ in range(n_pairs):
                query_text = cls.sample_sentences(batch["text"][idx], n_sentences)  # type: ignore
                key_idx = (
                    random.choice(list(same_auth_ids)) if same_auth_ids else local_idx
                )
                key = ds[key_idx]
                key_text = cls.sample_sentences(key["text"], n_sentences)

                # Append positive pair
                query_halids.append(batch["halid"][idx])  # type: ignore
                query_texts.append(query_text)
                query_years.append(batch["year"][idx])  # type: ignore
                query_domains.append(batch["domain"][idx])  # type: ignore
                query_affiliations.append(batch["affiliations"][idx])  # type: ignore
                query_authors.append(batch["authorids"][idx])  # type: ignore
                key_halids.append(key["halid"])
                key_texts.append(key_text)
                key_years.append(key["year"])
                key_domains.append(key["domain"])
                key_affiliations.append(key["affiliations"])
                key_authors.append(key["authorids"])
                domain_labels.append(
                    1 if set(batch["domain"][idx]) & set(key["domain"]) else 0  # type: ignore
                )
                affiliation_labels.append(
                    1
                    if set(batch["affiliations"][idx]) & set(key["affiliations"])  # type: ignore
                    else 0
                )
                author_labels.append(
                    1 if set(batch["authorids"][idx]) & set(key["authorids"]) else 0  # type: ignore
                )

            # Sample negative pairs
            for _ in range(n_pairs * 2):
                query_text = cls.sample_sentences(batch["text"][idx], n_sentences)  # type: ignore
                key_idx = random.choice(list(diff_auth_ids))
                key = ds[key_idx]
                key_text = cls.sample_sentences(key["text"], n_sentences)

                # Append negative pair
                query_halids.append(batch["halid"][idx])  # type: ignore
                query_texts.append(query_text)
                query_years.append(batch["year"][idx])  # type: ignore
                query_domains.append(batch["domain"][idx])  # type: ignore
                query_affiliations.append(batch["affiliations"][idx])  # type: ignore
                query_authors.append(batch["authorids"][idx])  # type: ignore
                key_halids.append(key["halid"])
                key_texts.append(key_text)
                key_years.append(key["year"])
                key_domains.append(key["domain"])
                key_affiliations.append(key["affiliations"])
                key_authors.append(key["authorids"])
                domain_labels.append(
                    1 if set(batch["domain"][idx]) & set(key["domain"]) else 0  # type: ignore
                )
                affiliation_labels.append(
                    1
                    if set(batch["affiliations"][idx]) & set(key["affiliations"])  # type: ignore
                    else 0
                )
                author_labels.append(
                    1 if set(batch["authorids"][idx]) & set(key["authorids"]) else 0  # type: ignore
                )

        return {
            "query_halid": query_halids,
            "query_text": query_texts,
            "query_year": query_years,
            "query_authors": query_authors,
            "query_affiliations": query_affiliations,
            "query_domains": query_domains,
            "key_halid": key_halids,
            "key_text": key_texts,
            "key_year": key_years,
            "key_authors": key_authors,
            "key_affiliations": key_affiliations,
            "key_domains": key_domains,
            "domain_label": domain_labels,
            "affiliation_label": affiliation_labels,
            "author_label": author_labels,
        }

    @staticmethod
    def sample_sentences(text: str, n_sentences: int):
        """Sample n sentences from a document."""
        sentences = sent_tokenize(text)
        sentence_idx = torch.randint(len(sentences), (1,)).item()
        while sentence_idx > len(sentences) - n_sentences:
            sentence_idx = torch.randint(len(sentences), (1,)).item()
        sentence = " ".join(sentences[sentence_idx : sentence_idx + n_sentences])

        return sentence
