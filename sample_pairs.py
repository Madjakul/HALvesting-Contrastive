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
        split="train[:1000]",
        streaming=config["ds"]["streaming"],
        cache_dir=args.cache_dir,
    )

    if config["sampler"]["do_sample"]:
        ContrastiveSampler.init_cache(ds)  # type: ignore

        augmented_ds = ds.map(
            ContrastiveSampler.sample_batched,
            batched=True,
            batch_size=config["sampler"]["batch_size"],
            with_indices=True,
            num_proc=args.num_proc if args.num_proc is not None else NUM_PROC,  # type: ignore
            remove_columns=ds.column_names,  # type: ignore
            fn_kwargs={
                "soft_positives": config["sampler"]["soft_positives"],
                "n_pairs": config["sampler"]["n_pairs"],
                "n_sentences": config["sampler"]["n_sentences"],
                "ds": ds,
                "all_ids": range(len(ds)),  # type: ignore
            },
            load_from_cache_file=config["sampler"]["load_from_cache_file"],  # type: ignore
        )

        config_name = "base-soft" if config["sampler"]["soft_positives"] else "base"

        if config["sampler"]["do_push_to_hub"]:
            augmented_ds.push_to_hub(  # type: ignore
                config["ds"]["push_checkpoint"],
                config_name=f"{config_name}-{config['sampler']['n_sentences']}",
            )

    if config["ict_sampler"]["do_sample"]:
        augmented_ds = ds.map(
            ICTSampler.sample_batched,
            batched=True,
            batch_size=config["ict_sampler"]["batch_size"],
            num_proc=args.num_proc if args.num_proc is not None else NUM_PROC,  # type: ignore
            remove_columns=ds.column_names,  # type: ignore
            fn_kwargs={
                "n_pairs": config["ict_sampler"]["n_pairs"],
                "n_sentences": config["ict_sampler"]["n_sentences"],
            },
            load_from_cache_file=config["ict_sampler"]["load_from_cache_file"],  # type: ignore
        )

        if config["ict_sampler"]["do_push_to_hub"]:
            augmented_ds.push_to_hub(  # type: ignore
                config["ds"]["push_checkpoint"],
                config_name=f"ict-{config['ict_sampler']['n_sentences']}",
            )
