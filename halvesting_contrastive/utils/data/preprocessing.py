# halvesting_contrastive/utils/data/preprocessing.py

import logging
from typing import Any, Dict, List

import datasets
import pandas as pd


class Preprocessing:
    """Preprocessing class for data preprocessing."""

    author_affiliations: pd.DataFrame
    author_papers: pd.DataFrame
    domains = (
        "shs",
        "info",
        "sdv",
        "spi",
        "phys",
        "math",
        "chim",
        "sde",
        "sdu",
        "scco",
        "stat",
        "qfin",
        "nlin",
    )

    @classmethod
    def batched_getsizeof(cls, documents: Dict[str, List[Any]]):
        """Get the size of the text in the documents.

        Parameters
        ----------
        documents: Dict[str, List[str]]
            Dictionary containing the documents.

        Returns
        -------
        documents:
        """
        size = []

        for text in documents["text"]:
            size.append(len(text))

        return {"size": size}

    @classmethod
    def batched_get_affiliations(cls, documents: Dict[str, List[Any]]):
        """Get the affiliation of the authors in the documents."""
        pass

    @classmethod
    def batched_get_authors(cls, documents: Dict[str, List[Any]]):
        """Get the authors in the documents."""
        pass

    @classmethod
    def batched_filter_domains(cls, documents: Dict[str, List[Any]]):
        """Filter the documents by domain."""
        pass
