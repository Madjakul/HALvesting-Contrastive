# halvesting_contrastive/utils/flusher.py

import json
import logging
import os
import os.path as osp
from typing import Any, Dict

from halvesting_contrastive.utils import helpers


class Flusher:
    """Class used to format documents and passages for contrastive learning.

    Parameters
    ----------
    output_dir : str
        Output directory for the formatted data.
    batch_size : int
        The number of documents to write to the output file before flushing.

    Attributes
    ----------
    output_dir : str
        Output directory for the formatted data.
    norm_output_dir : str
        Normalized output directory path.
    output_file : str
        Output file path.
    batch_size : int
        The number of documents to write to the output file before flushing.
    counter : int
        The number of batches processed.
    local_counter : int
        The number of documents written to the output file.
    """

    def __init__(self, output_dir: str, batch_size: int):
        self.output_dir = output_dir
        self.norm_output_dir = osp.normpath(output_dir)
        try:
            if self.norm_output_dir not in ["train", "dev", "test"]:
                raise ValueError(
                    f"Output directory must be 'train', 'dev', or 'test', "
                    f"not {self.norm_output_dir}."
                )
        except ValueError as e:
            logging.error(e)
        self.output_file = osp.join(
            self.output_dir, f"{self.norm_output_dir}_{self.counter}.jsonl"
        )
        self.batch_size = batch_size
        self.counter = self._get_counter()
        self.local_counter = 0

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        logging.info(f"Processed {self.counter} batches.")
        self.flush()
        self.local_counter = 0

    def _get_counter(self):
        checsum_path = osp.join(self.output_dir, "checksum.sha256")
        if osp.exists(checsum_path):
            with open(checsum_path, "rb") as f:
                counter = sum(1 for _ in f)
                return counter
        return 0

    def flush(self):
        """Compresses the output file, generates a checksum, and increments the
        counter."""
        gz_path = osp.join(self.output_dir, f"{self.output_file}.gz")
        helpers.gz_compress(gz_path)
        helpers.generate_checksum(dir_path=self.output_dir, gz_file_path=gz_path)
        os.remove(self.output_file)
        self.counter += 1
        self.local_counter = 0
        self.output_file = osp.join(
            self.output_dir, f"{self.norm_output_dir}_{self.counter}.jsonl"
        )

    def write(self, data: Dict[str, Any]):
        """Writes the data to the output file.

        Parameters
        ----------
        data : Dict[str, Any]
            Data to be written to the output file.
        """
        with open(self.output_file, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        self.local_counter += 1
        if self.local_counter == self.batch_size:
            self.flush()
