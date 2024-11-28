# halvesting_contrastive/core/passage_sampler.py

import logging
import multiprocessing as mp
import random
import re
from typing import Any, Dict

import datasets
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from halvesting_contrastive.core.formater import Formater
from halvesting_contrastive.utils import helpers


class PassageSampler:
    """Class used to sample documents and grammatically correct passages from
    documents."""

    def __init__(
        self,
        dataset: datasets.Dataset,
        output_dir: str,
        num_proc: int,
        num_pairs: int,
        alpha: float = 0.5,
    ):
        self.sampling_methods = [self.sample_paragraph, self.sample_sentence]
        self.dataset = dataset
        try:
            assert num_proc > 1
        except AssertionError:
            raise ValueError("num_proc must be greater than 1.")
        self.num_proc = num_proc
        logging.info(f"Using {self.num_proc} processes.")
        self.num_pairs = num_pairs
        self.output_dir = helpers.check_dir(output_dir)
        sizes = torch.FloatTensor(dataset["size"])
        self.probs = self._compute_multinomial_probs(sizes, alpha)

    def __call__(self):
        queue = mp.Queue()

        def formatter_worker(queue):
            """Process results from the queue and save them using the
            Formatter."""
            with Formater(self.output_dir, 1000, 100000) as formatter:
                while True:
                    result = queue.get()
                    if result is None:
                        break
                    formatter.save(
                        query=result["query"],
                        query_is_paragraph=result["query_is_paragraph"],
                        query_text=result["query_text"],
                        passage=result["passage"],
                        passage_is_paragraph=result["passage_is_paragraph"],
                        passage_text=result["passage_text"],
                    )

        # Start the formatter process
        formatter_process = mp.Process(target=formatter_worker, args=(queue,))
        formatter_process.start()

        # Launch workers to sample pairs
        with mp.Pool(self.num_proc - 1) as pool:
            for _ in tqdm(range(self.num_pairs)):
                pool.apply_async(
                    self.sample_pairs,
                    args=(),
                    callback=lambda result: queue.put(result),
                )
            pool.close()
            pool.join()

        # Signal the formatter process to terminate
        queue.put(None)
        formatter_process.join()

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
        # TODO: remove this function as it useless
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
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        paragraph_idx = torch.randint(len(paragraphs), (1,)).item()
        paragraph = paragraphs[paragraph_idx]
        return True, paragraph

    def sample_sentence(self, document: Dict[str, Any]):
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
        return False, sentence

    def sample_pairs(self):
        query_idx, passage_idx = self.sample_documents(2)
        query_idx = query_idx.item()
        passage_idx = passage_idx.item()
        query = self.dataset[query_idx]
        passage = self.dataset[passage_idx]

        query_is_paragraph, query_text = self.sampling_methods[random.randint(0, 1)](
            query
        )
        passage_is_paragraph, passage_text = self.sampling_methods[
            random.randint(0, 1)
        ](passage)

        return {
            "query": query,
            "query_is_paragraph": query_is_paragraph,
            "query_text": query_text,
            "passage": passage,
            "passage_is_paragraph": passage_is_paragraph,
            "passage_text": passage_text,
        }

    @staticmethod
    def _compute_multinomial_probs(sizes: torch.FloatTensor, alpha: float):
        regularizer = torch.sum(sizes**alpha)
        probs = sizes**alpha / regularizer
        print(probs[:100])
        return probs
