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
    """Class used to sample documents and passages for contrastive learning."""

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
        passage_halids = []
        passage_texts = []
        passage_years = []
        passage_domains = []
        passage_affiliations = []
        passage_authors = []
        domain_labels = []
        affiliation_labels = []
        author_labels = []

        for idx in range(len(batch)):
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
                passage_idx = (
                    random.choice(list(same_auth_ids)) if same_auth_ids else local_idx
                )
                passage = ds[passage_idx]
                passage_text = cls.sample_sentences(passage["text"], n_sentences)

                # Append positive pair
                query_halids.append(batch["halid"][idx])  # type: ignore
                query_texts.append(query_text)
                query_years.append(batch["year"][idx])  # type: ignore
                query_domains.append(batch["domain"][idx])  # type: ignore
                query_affiliations.append(batch["affiliations"][idx])  # type: ignore
                query_authors.append(batch["authorids"][idx])  # type: ignore
                passage_halids.append(passage["halid"])
                passage_texts.append(passage_text)
                passage_years.append(passage["year"])
                passage_domains.append(passage["domain"])
                passage_affiliations.append(passage["affiliations"])
                passage_authors.append(passage["authorids"])
                domain_labels.append(
                    1 if set(batch["domain"][idx]) & set(passage["domain"]) else 0  # type: ignore
                )
                affiliation_labels.append(
                    1
                    if set(batch["affiliations"][idx]) & set(passage["affiliations"])  # type: ignore
                    else 0
                )
                author_labels.append(
                    1 if set(batch["authorids"][idx]) & set(passage["authorids"]) else 0  # type: ignore
                )

            # Sample negative pairs
            for _ in range(n_pairs * 2):
                query_text = cls.sample_sentences(batch["text"][idx], n_sentences)  # type: ignore
                passage_idx = random.choice(list(diff_auth_ids))
                passage = ds[passage_idx]
                passage_text = cls.sample_sentences(passage["text"], n_sentences)

                # Append negative pair
                query_halids.append(batch["halid"][idx])  # type: ignore
                query_texts.append(query_text)
                query_years.append(batch["year"][idx])  # type: ignore
                query_domains.append(batch["domain"][idx])  # type: ignore
                query_affiliations.append(batch["affiliations"][idx])  # type: ignore
                query_authors.append(batch["authorids"][idx])  # type: ignore
                passage_halids.append(passage["halid"])
                passage_texts.append(passage_text)
                passage_years.append(passage["year"])
                passage_domains.append(passage["domain"])
                passage_affiliations.append(passage["affiliations"])
                passage_authors.append(passage["authorids"])
                domain_labels.append(
                    1 if set(batch["domain"][idx]) & set(passage["domain"]) else 0  # type: ignore
                )
                affiliation_labels.append(
                    1
                    if set(batch["affiliations"][idx]) & set(passage["affiliations"])  # type: ignore
                    else 0
                )
                author_labels.append(
                    1 if set(batch["authorids"][idx]) & set(passage["authorids"]) else 0  # type: ignore
                )

        print(f"Processed {len(batch)} documents, generated {len(query_halids)} pairs")
        return {
            "query_halid": query_halids,
            "query_text": query_texts,
            "query_year": query_years,
            "query_authors": query_authors,
            "query_affiliations": query_affiliations,
            "query_domains": query_domains,
            "passage_halid": passage_halids,
            "passage_text": passage_texts,
            "passage_year": passage_years,
            "passage_authors": passage_authors,
            "passage_affiliations": passage_affiliations,
            "passage_domains": passage_domains,
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
