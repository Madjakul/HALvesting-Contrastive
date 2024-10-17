# halvesting_contrastive/utils/data/preprocessing.py

import logging
from typing import Any, Dict, List


class Preprocessing:
    """Preprocessing class for data preprocessing."""

    metadata: Dict[str, Any] = {}
    domain_list = (
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
    def set_metadata(cls, metadata: Dict[str, Any]):
        """Set the metadata."""
        cls.metadata = metadata

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
    def batched_get_authors(cls, documents: Dict[str, List[Any]]):
        """Get the authors in the documents."""
        if not cls.metadata:
            raise ValueError("Metadata is not set.")
        authorids = []
        affiliations = []
        for halid in documents["halid"]:
            local_metadata = cls.metadata[halid]
            local_authorids = []
            local_affiliations = []
            for author in local_metadata["authors"]:
                id = author["halauthorid"]
                if id != "0":
                    local_authorids.append(id)
                local_affiliations.extend(author["affiliations"])
            authorids.append(local_authorids)
            affiliations.append(local_affiliations)
        return {"authorids": authorids, "affiliations": affiliations}

    @classmethod
    def batched_filter_domains(cls, documents: Dict[str, List[Any]]):
        """Filter the documents by domain."""
        domains = []
        for local_domains in documents["domain"]:
            filtered_domains = []
            for domain in local_domains:
                filtered_domain = domain.split(".")[0]
                if filtered_domain not in cls.domain_list:
                    continue
                filtered_domains.append(filtered_domain)
            domains.append(list(set(filtered_domains)))
        return {"domain": domains}
