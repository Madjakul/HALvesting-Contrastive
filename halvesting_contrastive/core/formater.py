# halvesting_contrastive/core/formater.py

import json
import os.path as osp
from datetime import datetime
from typing import Any, Dict

from halvesting_contrastive.utils import helpers


# TODO: document  the class
class Formater:
    """Class used to format and write the query, passage, and output to a
    file."""

    def __init__(self, output_dir: str, batch_size: int, doc_size: int):
        try:
            assert batch_size > 0
        except AssertionError:
            raise ValueError("batch_size must be greater than 0.")
        try:
            assert batch_size < doc_size
        except AssertionError:
            raise ValueError("batch_size must be less than doc_counter.")
        self.output_dir = helpers.check_dir(output_dir)
        self.batch_size = batch_size
        self.doc_size = doc_size
        self.counter = 0
        self.doc_counter = 0
        self.batch = []
        self.now = datetime.now().strftime("%Y-%m-%d")

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.flush()

    def format(
        self,
        query: Dict[str, Any],
        query_text: str,
        passage: Dict[str, Any],
        passage_text: str,
    ):
        query_halid = query["halid"]
        query_year = query["year"]
        query_authors = query["authorids"]
        query_domains = query["domain"]
        query_affiliations = query["affiliations"]
        passage_halid = passage["halid"]
        passage_year = passage["year"]
        passage_authors = passage["authorids"]
        passage_domains = passage["domain"]
        passage_affiliations = passage["affiliations"]
        year_label = 1 if query_year == passage_year else 0
        domain_label = 1 if set(query_domains) & set(passage_domains) else 0
        author_label = 1 if set(query_authors) & set(passage_authors) else 0
        affiliation_label = (
            1 if set(query_affiliations) & set(passage_affiliations) else 0
        )
        output = {
            "query_halid": query_halid,
            "query_text": query_text,
            "query_year": query_year,
            "query_authors": query_authors,
            "query_affiliations": query_affiliations,
            "query_domains": query_domains,
            "passage_halid": passage_halid,
            "passage_text": passage_text,
            "passage_year": passage_year,
            "passage_authors": passage_authors,
            "passage_affiliations": passage_affiliations,
            "passage_domains": passage_domains,
            "year_label": year_label,
            "domain_label": domain_label,
            "affiliation_label": affiliation_label,
            "author_label": author_label,
        }
        return output

    def save(
        self,
        query: Dict[str, Any],
        query_text: str,
        passage: Dict[str, Any],
        passage_text: str,
    ):
        self.batch.append(
            self.format(
                query,
                query_text,
                passage,
                passage_text,
            )
        )
        if len(self.batch) >= self.batch_size:
            self.flush()

    def flush(self):
        path = osp.join(self.output_dir, f"pairs_{self.now}_{self.counter}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for pair in self.batch:
                json.dump(pair, f, ensure_ascii=False)
                f.write("\n")

        self.doc_counter += len(self.batch)
        self.batch.clear()

        if self.doc_counter >= self.doc_size:
            self.counter += 1
            self.doc_counter = 0
