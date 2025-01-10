# halvesting_contrastive/core/passage_sampler.py

import logging
import multiprocessing as mp
import os
import random
from typing import Any, Dict

import datasets
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from halvesting_contrastive.core.formater import Formater
from halvesting_contrastive.utils import helpers

_worker_seeded: bool = False


class PassageSampler:
    # TODO: test the class on edge cases
    # TODO: document the class
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

    def __call__(self, num_sentences: int):
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
                        query_text=result["query_text"],
                        passage=result["passage"],
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
                    args=(num_sentences,),
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

    def sample_sentence(self, document: Dict[str, Any], num_sentences: int = 1):
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
        while sentence_idx > len(sentences) - num_sentences:
            sentence_idx = torch.randint(len(sentences), (1,)).item()
        sentence = " ".join(sentences[sentence_idx : sentence_idx + num_sentences])

        return sentence

    def sample_pairs(self, num_sentences: int):
        # Set a unique seed for this process based on its PID
        global _worker_seeded
        if not _worker_seeded:
            process_id = os.getpid()
            random.seed(process_id)
            torch.manual_seed(process_id)
            _worker_seeded = True

        query_idx, passage_idx = self.sample_documents(2)
        query_idx = query_idx.item()
        passage_idx = passage_idx.item()
        query = self.dataset[query_idx]
        passage = self.dataset[passage_idx]

        query_text = self.sample_sentence(query, num_sentences)
        passage_text = self.sample_sentence(passage, num_sentences)

        return {
            "query": query,
            "query_text": query_text,
            "passage": passage,
            "passage_text": passage_text,
        }

    @staticmethod
    def _compute_multinomial_probs(sizes: torch.FloatTensor, alpha: float):
        regularizer = torch.sum(sizes**alpha)
        probs = sizes**alpha / regularizer
        return probs
