# sample_pairs.py

import logging

import datasets
import nltk
import psutil

from halvesting_contrastive.core import PassageSampler
from halvesting_contrastive.utils import helpers, logging_config
from halvesting_contrastive.utils.argparsers import SamplerArgparse

NUM_PROC = psutil.cpu_count(logical=False)


logging_config()
nltk.download("punkt")
nltk.download("punkt_tab")


if __name__ == "__main__":
    args = SamplerArgparse.parse_known_args()

    logging.info(f"{'=' * helpers.WIDTH}")
    logging.info("Sampling passages".center(helpers.WIDTH))
    logging.info(f"{'=' * helpers.WIDTH}")

    config = helpers.load_config_from_file(args.config_path)

    ds = datasets.load_dataset(
        config["ds"]["checkpoint"],
        split="train",
        streaming=config["ds"]["streaming"],
        cache_dir=config["ds"]["cache_dir"] if "cache_dir" in config["ds"] else None,
    )

    sampler = PassageSampler(
        dataset=ds,  # type: ignore
        output_dir=args.output_dir,
        num_proc=args.num_proc if args.num_proc is not None else NUM_PROC,
        num_pairs=config["sampler"]["num_pairs"],
        alpha=config["sampler"]["alpha"],
    )
    sampler(config["sampler"]["num_sentences"])

    if config["main"]["do_checksum"]:
        # TODO: implement checksum
        pass

    if config["main"]["do_push_to_hub"]:
        # TODO: implement push_to_hub
        ds.push_to_hub(config["main"]["checkpoint"], private=True)
