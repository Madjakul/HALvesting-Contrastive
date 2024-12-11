# preprocess.py

import json
import logging
import os
import os.path as osp

import datasets
import psutil

from halvesting_contrastive.utils import helpers, logging_config
from halvesting_contrastive.utils.argparsers import PreprocessArgparse
from halvesting_contrastive.utils.data import Preprocessing

NUM_PROC = psutil.cpu_count(logical=False)
logging_config()


if __name__ == "__main__":
    args = PreprocessArgparse.parse_known_args()
    config = helpers.load_config_from_file(args.config_path)

    logging.info(f"{('=' * helpers.WIDTH)}")
    logging.info(f"Preprocessing HALvest".center(helpers.WIDTH))
    logging.info(f"{('=' * helpers.WIDTH)}")

    # Load metadata
    if args.do_convert_responses:
        logging.info(f"Converting responses in {args.responses_dir} to metadata.")
        response_files = [
            osp.join(args.responses_dir, f)
            for f in os.listdir(args.responses_dir)
            if f.endswith(".json")
        ]
        metadata = helpers.jsons_to_dict(response_files, on="halid")
        with open(args.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    else:
        logging.info(f"Loading metadata from {args.metadata_path}.")
        with open(args.metadata_path, "r") as f:
            metadata = json.load(f)

    logging.info(f"Loading dataset from {config['ds']['checkpoint']}.")
    ds = datasets.load_dataset(
        config["ds"]["checkpoint"], config["ds"]["config"], split="train"
    )

    # Preprocessing dataset
    Preprocessing.set_metadata(metadata)
    logging.info("Filtering documents with unknown domains.")
    ds = ds.map(
        lambda batch: Preprocessing.batched_filter_domains(batch),
        batched=True,
        batch_size=config["map"]["batch_size"],
        num_proc=args.num_proc if args.num_proc else NUM_PROC,  # type: ignore
        load_from_cache_file=config["map"]["load_from_cache_file"],  # type: ignore
    )
    ds = ds.filter(lambda document: len(document["domain"]) > 0)
    logging.info("Getting the size of the documents.")
    size = ds.map(
        lambda batch: Preprocessing.batched_getsizeof(batch),
        batched=True,
        batch_size=config["map"]["batch_size"],
        num_proc=args.num_proc if args.num_proc else NUM_PROC,
        load_from_cache_file=config["map"]["load_from_cache_file"],
    )
    ds = ds.add_column("size", size["size"])  # type: ignore
    logging.info("Getting the authors and affiliations.")
    authors = ds.map(
        lambda batch: Preprocessing.batched_get_authors(batch),
        batched=True,
        batch_size=config["map"]["batch_size"],
        num_proc=args.num_proc if args.num_proc else NUM_PROC,
        load_from_cache_file=config["map"]["load_from_cache_file"],
    )
    ds = ds.add_column("authorids", authors["authorids"])
    ds = ds.add_column("affiliations", authors["affiliations"])

    logging.info("FIltering out documents with no authors...")
    ds = ds.filter(lambda document: len(document["authorids"]) > 0)
    logging.info("Filtering out documents with no affiliations...")
    ds = ds.filter(lambda document: len(document["affiliations"]) > 0)
    logging.info("Filtering out documents with unknown domains...")
    ds = ds.filter(
        lambda document: set(document["domain"]).issubset(
            set(Preprocessing.domain_list)
        )
    )

    if config["main"]["do_push_to_hub"]:
        ds.push_to_hub(config["main"]["checkpoint"], private=True)
