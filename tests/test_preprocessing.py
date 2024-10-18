# tests/test_preprocessing.py

import datasets
import pytest

from halvesting_contrastive.utils import helpers
from halvesting_contrastive.utils.data import Preprocessing


@pytest.fixture
def dataset():
    return datasets.load_dataset("json", data_files="./data/mock/mock_halvest.jsonl")


def test_batched_getsizeof(dataset):
    documents = dataset["train"]
    size = documents.map(
        lambda batch: (Preprocessing.batched_getsizeof(batch)),
        batched=True,
        batch_size=1,
        num_proc=2,
        load_from_cache_file=False,
    )
    documents = documents.add_column("size", size["size"])
    assert "size" in documents.column_names
    for document in documents:
        assert document["size"] == len(document["text"])


def test_batched_filter_domains(dataset):
    documents = dataset["train"]
    documents = documents.map(
        lambda batch: Preprocessing.batched_filter_domains(batch),
        batched=True,
        batch_size=1,
        num_proc=2,
        load_from_cache_file=False,
    )
    for document in documents:
        for domain in document["domain"]:
            assert domain in Preprocessing.domain_list


def test_batched_get_authors(dataset):
    documents = dataset["train"]
    metadata = helpers.json_to_dict("./data/mock/mock_reponses.json", on="halid")
    Preprocessing.set_metadata(metadata)
    authors = documents.map(
        lambda batch: Preprocessing.batched_get_authors(batch),
        batched=True,
        batch_size=1,
        num_proc=2,
        load_from_cache_file=False,
    )
    documents = documents.add_column("authorids", authors["authorids"])
    documents = documents.add_column("affiliations", authors["affiliations"])
    assert "authorids" in documents.column_names
    assert "affiliations" in documents.column_names
    for document in documents:
        halid = document["halid"]
        stored_authorids = []
        stored_affiliations = []
        for author in Preprocessing.metadata[halid]["authors"]:
            stored_authorids.append(author["halauthorid"])
            stored_affiliations.extend(author["affiliations"])
        for authorid in document["authorids"]:
            assert authorid in stored_authorids
        for affiliation in document["affiliations"]:
            assert affiliation in stored_affiliations
