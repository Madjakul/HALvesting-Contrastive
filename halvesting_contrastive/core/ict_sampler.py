# halvesting_contrastive/core/ict_sampler.py

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import nltk

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


class ICTSampler:
    """Class used to sample documents and create triplets for Inverse Cloze
    Task (ICT).

    For each document, it samples an anchor passage, a positive context, and a
    hard negative.

    - The positive context is the sentences surrounding the anchor. Following the
      original paper, 90% of the time the anchor text is removed from this
      context, and 10% of the time it is kept.
    - The hard negative is another, non-overlapping passage from the same document.
    """

    @classmethod
    def sample_batched(
        cls, batch: Dict[str, List], n_triplets: int, n_sentences: int = 1
    ):
        """Sample triplets from the same document in a batch.

        Parameters
        ----------
        batch: List[Dict[str, Any]]
            Batch of documents.
        n_triplets: int
            Number of triplets to sample per document.
        n_sentences: int
            Number of contiguous sentences to sample for the anchor.

        Returns
        -------
        Dict[str, List[Any]]
            Dictionary containing the anchor, positive, and negative texts,
            along with anchor metadata.
        """
        results = defaultdict(list)

        for idx in range(len(batch["text"])):  # type: ignore
            for _ in range(n_triplets):
                anchor_text, positive_text, negative_text = cls._sample_triplet(
                    batch["text"][idx], n_sentences  # type: ignore
                )

                if not all([anchor_text, positive_text, negative_text]):
                    continue

                results["halid"].append(batch["halid"][idx])  # type: ignore
                results["year"].append(batch["year"][idx])  # type: ignore
                results["domains"].append(batch["domain"][idx])  # type: ignore
                results["affiliations"].append(batch["affiliations"][idx])  # type: ignore
                results["authors"].append(batch["authorids"][idx])  # type: ignore
                results["query"].append(anchor_text)
                results["positive"].append(positive_text)
                results["negative"].append(negative_text)
        return results

    @staticmethod
    def _sample_triplet(text: str, n_sentences: int) -> Tuple[str, str, str]:
        """Samples an anchor, its context (positive), and a random non-
        overlapping passage (negative) from the text.

        Parameters
        ----------
        text: str
            Text from which to sample.
        n_sentences: int
            Number of sentences for the anchor.

        Returns
        -------
        Tuple[str, str, str]
            A tuple of (anchor, positive, negative) texts. Returns empty strings
            if sampling is not possible.
        """
        sentences = tokenize_and_merge_sentences(text)
        # The positive context is 2*n sentences, the negative is also 2*n sentences.
        context_size = 2 * n_sentences

        # Min sentences: 3*n for anchor+context block, plus 2*n for the negative block.
        min_required_sentences = (3 * n_sentences) + context_size
        if len(sentences) < min_required_sentences:
            return "", "", ""

        # 1. Sample anchor and define context boundaries
        query_start_range = n_sentences
        query_end_range = len(sentences) - context_size
        if query_start_range >= query_end_range:
            return "", "", ""
        query_idx = random.randint(query_start_range, query_end_range)

        anchor = " ".join(sentences[query_idx : query_idx + n_sentences])

        # 2. Probabilistically define the positive context (90/10 split)
        if random.random() < 0.1:
            # 10% OF THE TIME: Positive context INCLUDES the anchor (vocabulary matching)
            # The key is the whole original block of 3*n_sentences.
            positive_sentences = sentences[
                query_idx - n_sentences : query_idx + context_size
            ]
        else:
            # 90% OF THE TIME: Positive context EXCLUDES the anchor (true ICT)
            # The key is the surrounding context only.
            positive_sentences = (
                sentences[query_idx - n_sentences : query_idx]
                + sentences[query_idx + n_sentences : query_idx + context_size]
            )
        positive = " ".join(positive_sentences)

        # 3. Sample a non-overlapping hard negative passage
        forbidden_start = query_idx - n_sentences
        forbidden_end = query_idx + context_size

        possible_neg_starts = []
        # Add possible start indices for a negative block before the forbidden zone
        for i in range(forbidden_start - context_size + 1):
            possible_neg_starts.append(i)
        # Add possible start indices for a negative block after the forbidden zone
        for i in range(forbidden_end, len(sentences) - context_size + 1):
            possible_neg_starts.append(i)

        if not possible_neg_starts:
            return "", "", ""

        neg_idx = random.choice(possible_neg_starts)
        negative = " ".join(sentences[neg_idx : neg_idx + context_size])

        return anchor, positive, negative
