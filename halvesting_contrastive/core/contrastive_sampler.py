# halvesting_contrastive/utils/data/sampler.py

import logging
import random
from collections import defaultdict
from typing import Dict, List, Set

import datasets
import nltk
from tqdm import tqdm

# Ensure the sentence tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")


MERGE_TRIGGERS = (
    "et al.",
    "e.g.",
    "i.e.",
    "vs.",
    "cf.",
    "etc.",
    "approx.",
    "fig.",
    "figs.",
    "tab.",
    "sec.",
    "chap.",
    "dr.",
    "Mr.",
    "mrs.",
    "ms.",
    "prof.",
    "p.",
    "pp.",
    "vol.",
    "no.",
    "a.d.",
    "b.c.",
)


def tokenize_and_merge_sentences(text: str) -> list[str]:
    """Tokenizes text into sentences and merges sentences that end with
    specific abbreviations."""
    # Initial tokenization using NLTK
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    merged_sentences = []
    i = 0
    while i < len(sentences):
        current_sentence = sentences[i]
        # Check if the sentence ends with a trigger and is not the last sentence
        while i + 1 < len(sentences) and current_sentence.lower().strip().endswith(
            MERGE_TRIGGERS
        ):
            # Merge with the next sentence
            current_sentence += " " + sentences[i + 1]
            i += 1
        merged_sentences.append(current_sentence)
        i += 1

    return merged_sentences


class ContrastiveSampler:

    auth_to_idx: Dict[Set, List[int]]
    auth_to_idx_soft: Dict[str, List[int]]
    domain_to_idx: Dict[Set, List[int]]

    @classmethod
    def init_cache(cls, ds: datasets.Dataset):
        logging.info("Building author to indices map...")
        cls.auth_to_idx = cls._auth_to_idx(ds)
        logging.info("Building author to indices map (soft)...")
        cls.auth_to_idx_soft = cls._auth_to_idx_soft(ds)
        logging.info("Building domain to indices map...")
        cls.domain_to_idx = cls._domain_to_idx(ds)

    @classmethod
    def generate_triplet_candidates(
        cls,
        batch: Dict[str, List],
        ids: List[int],
        ds: datasets.Dataset,
        n_triplets: int,
        n_sentences: int,
        all_ids: List[int],
    ):
        all_pool = set(all_ids)
        query_halids = []
        query_texts = []
        query_years = []
        query_domains = []
        query_affiliations = []
        query_authorids = []
        pos_halids = []
        pos_texts = []
        pos_years = []
        pos_domains = []
        pos_affiliations = []
        pos_authorids = []
        neg_halids = []
        neg_texts = []
        neg_years = []
        neg_domains = []
        neg_affiliations = []
        neg_authorids = []

        for idx, query in enumerate(tqdm(batch["text"])):
            global_idx = ids[idx]
            neg_pool = cls.get_neg_pool(
                batch["authorids"][idx], batch["domain"][idx], all_pool
            )
            if len(neg_pool) == 0:
                continue
            pos_pool = cls.get_pos_pool(batch["authorids"][idx], global_idx)
            if len(pos_pool) == 0:
                continue
            pos_idx, neg_idx = cls.sample_triplets(pos_pool, neg_pool, n_triplets)
            for p_idx, n_idx in zip(pos_idx, neg_idx):
                for i in range(n_triplets):
                    query_halids.append(batch["halid"][idx])
                    query_texts.append(cls._get_random_span(query, n_sentences))
                    query_years.append(batch["year"][idx])
                    query_domains.append(batch["domain"][idx])
                    query_affiliations.append(batch["affiliations"][idx])
                    query_authorids.append(batch["authorids"][idx])
                    pos_halids.append(ds[p_idx]["halid"])
                    pos_texts.append(
                        cls._get_random_span(ds[p_idx]["text"], n_sentences)
                    )
                    pos_years.append(ds[p_idx]["year"])
                    pos_domains.append(ds[p_idx]["domain"])
                    pos_affiliations.append(ds[p_idx]["affiliations"])
                    pos_authorids.append(ds[p_idx]["authorids"])
                    neg_halids.append(ds[n_idx[i]]["halid"])
                    neg_texts.append(
                        cls._get_random_span(ds[n_idx[i]]["text"], n_sentences)
                    )
                    neg_years.append(ds[n_idx[i]]["year"])
                    neg_domains.append(ds[n_idx[i]]["domain"])
                    neg_affiliations.append(ds[n_idx[i]]["affiliations"])
                    neg_authorids.append(ds[n_idx[i]]["authorids"])
        return {
            "query_halid": query_halids,
            "query": query_texts,
            "query_year": query_years,
            "query_domain": query_domains,
            "query_affiliations": query_affiliations,
            "query_authorids": query_authorids,
            "pos_halid": pos_halids,
            "positive": pos_texts,
            "pos_year": pos_years,
            "pos_domain": pos_domains,
            "pos_affiliations": pos_affiliations,
            "pos_authorids": pos_authorids,
            "neg_halids": neg_halids,
            "negative": neg_texts,
            "neg_year": neg_years,
            "neg_domain": neg_domains,
            "neg_affiliations": neg_affiliations,
            "neg_authorids": neg_authorids,
        }

    @classmethod
    def get_pos_pool(cls, auths: List[str], global_idx: int):
        pos_pool = set(cls.auth_to_idx[frozenset(set(auths))])
        pos_pool -= {global_idx}  # ensure not the same document
        # pos_pool |= {global_idx} # test without the same document
        return list(pos_pool)

    @classmethod
    def get_neg_pool(cls, auths: List[str], domains: List[str], all_pool: Set[int]):
        soft_pos_pool = set()
        for auth in auths:
            soft_pos_pool |= set(cls.auth_to_idx_soft[auth])
        neg_pool = all_pool - soft_pos_pool
        neg_pool &= set(cls.domain_to_idx.get(frozenset(set(domains)), []))
        return list(neg_pool)

    @staticmethod
    def sample_triplets(pos_pool: List[int], neg_pool: List[int], n_triplets: int):
        pos_idx = random.choices(pos_pool, k=n_triplets)
        neg_idx = random.choices(neg_pool, k=(n_triplets * n_triplets))
        neg_idx = [
            neg_idx[i : i + n_triplets] for i in range(0, len(neg_idx), n_triplets)
        ]
        return pos_idx, neg_idx

    @staticmethod
    def _auth_to_idx(ds: datasets.Dataset):
        d = defaultdict(list)
        for idx, auths in enumerate(tqdm(ds["authorids"])):
            d[frozenset(set(auths))].append(idx)
        return d

    @staticmethod
    def _auth_to_idx_soft(ds: datasets.Dataset):
        d = defaultdict(list)
        for idx, auths in enumerate(tqdm(ds["authorids"])):
            for auth in auths:
                d[auth].append(idx)
        return d

    @staticmethod
    def _domain_to_idx(ds: datasets.Dataset):
        d = defaultdict(list)
        for idx, domains in enumerate(tqdm(ds["domain"])):
            d[frozenset(set(domains))].append(idx)
        return d

    @staticmethod
    def _get_random_span(text: str, n_sentences: int) -> str:
        sentences = tokenize_and_merge_sentences(text)
        if len(sentences) <= n_sentences:
            return text

        # Directly calculate the valid range for the starting index and select one.
        # This avoids errors and inefficient rejection sampling.
        start_index = random.randint(0, len(sentences) - n_sentences)
        return " ".join(sentences[start_index : start_index + n_sentences])
