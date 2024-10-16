# tests/test_preprocessing.py

import datasets
import pytest

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
    )
    documents = documents.add_column("size", size["size"])
    assert "size" in documents.column_names
    for document in documents:
        assert document["size"] == len(document["text"])
