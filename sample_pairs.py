# sample_pairs.py

import logging

import datasets
import nltk
import psutil

from halvesting_contrastive.core import ContrastiveSampler, ICTSampler
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

    # TODO: make this sampling method option
    ContrastiveSampler.init_cache(ds)

    augmented_ds = ds.map(
        ContrastiveSampler.sample_batched,
        batched=True,
        batch_size=config["sampler"]["batch_size"],
        with_indices=True,
        num_proc=args.num_proc if args.num_proc is not None else NUM_PROC,
        remove_columns=ds.column_names,
        fn_kwargs={
            "soft_positives": config["sampler"]["soft_positives"],
            "n_pairs": config["sampler"]["n_pairs"],
            "n_sentences": config["sampler"]["n_sentences"],
            "ds": ds,
            "all_ids": range(len(ds)),
        },
    )

    # TODO: Implement the ICTSampler option

    if config["main"]["do_checksum"]:
        # TODO: implement checksum
        pass

    if config["main"]["do_push_to_hub"]:
        # TODO: implement push_to_hub
        ds.push_to_hub(config["main"]["checkpoint"], private=True)
