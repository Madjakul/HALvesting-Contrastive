# halvesting_contrastive/core/passage_sampler.py

import random
from typing import Any, Dict

import datasets
import torch
from nltk.tokenize import sent_tokenize


class PassageSampler:
    """Class used to sample documents and grammatically correct passages from
    documents."""

    def __init__(self, dataset: datasets.Dataset, alpha: float = 0.5):
        self.sampling_methods = [self.sample_paragraph, self.sample_sentences]
        self.dataset = dataset
        sizes = torch.FloatTensor(dataset["size"], alpha)
        self.probs = self._compute_multinomial_probs(sizes, alpha)

    def sample_documents(self, batch_size: int):
        """Sample a batch of document indices given the multinomial
        distribution over the dataset sizes [1]_.

        Parameters
        ----------
        batch_size : int
            The number of documents to sample.

        Returns
        -------
        idx : torch.Tensor
            The sampled document indices.

        References
        ----------
        ..  [1] Li, Zehan, et al. Towards General Text Embeddings with Multi-Stage
            Contrastive Learning. arXiv:2308.03281, arXiv, 6 Aug. 2023. arXiv.org,
            https://doi.org/10.48550/arXiv.2308.03281.
        """
        idx = torch.multinomial(self.probs, num_samples=batch_size, replacement=True)
        return idx

    def sample_paragraph(self, document: Dict[str, Any]):
        """Sample a paragraph from a document.

        Parameters
        ----------
        document : Dict[str, Any]
            A document from the dataset.

        Returns
        -------
        paragraph : str
            A paragraph from the document.
        """
        text = document["text"]
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        paragraph_idx = torch.randint(len(paragraphs), (1,)).item()
        paragraph = paragraphs[paragraph_idx]
        return paragraph

    def sample_sentences(self, document: Dict[str, Any]):
        """Sample a sentence from a document.

        Parameters
        ----------
        document : Dict[str, Any]
            A document from the dataset.

        Returns
        -------
        sentence : str
            A sentence from the document
        """
        sentences = sent_tokenize(document["text"])
        sentence_idx = torch.randint(len(sentences), (1,)).item()
        sentence = sentences[sentence_idx]
        return sentence

    def sample_pairs(self):
        query_idx, passage_idx = self.sample_documents(2)
        query_idx = query_idx.item()
        passage_idx = passage_idx.item()
        query = self.dataset[query_idx]
        passage = self.dataset[passage_idx]

        query_text = self.sampling_methods[random.randint(0, 1)](query)
        passage_text = self.sampling_methods[random.randint(0, 1)](passage)

        # TODO: Format pairs and write them into a jsonl file

    @staticmethod
    def _compute_multinomial_probs(sizes: torch.FloatTensor, alpha: float):
        regularizer = torch.sum(sizes**alpha)
        probs = sizes**alpha / regularizer
        return probs
