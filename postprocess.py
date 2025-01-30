# postprocess.py

import logging

import datasets
import psutil

from halvesting_contrastive.utils import helpers, logging_config
from halvesting_contrastive.utils.argparsers import PostprocessArgparse
from halvesting_contrastive.utils.data import Postprocessing

NUM_PROC = psutil.cpu_count(logical=False)
logging_config()


if __name__ == "__main__":
    args = PostprocessArgparse.parse_known_args()
    config = helpers.load_config_from_file(args.config_path)

    logging.info(f"{('=' * helpers.WIDTH)}")
    logging.info(f"Postprocessing HALvest".center(helpers.WIDTH))
    logging.info(f"{('=' * helpers.WIDTH)}")

    logging.info(f"Loading dataset from {config['ds']['checkpoint']}.")
    ds = datasets.load_dataset(
        config["ds"]["checkpoint"], config["ds"]["config"], split="train"
    )

    logging.info("Filtering documents...")
    ds = ds.filter(
        Postprocessing.run,
        num_proc=args.num_proc if args.num_proc else NUM_PROC,
        load_from_cache_file=config["filter"]["load_from_cache_file"],
    )

    if config["main"]["do_push_to_hub"]:
        logging.info("Splitting the dataset...")
        train_testvalid = ds.train_test_split(test_size=0.2)
        test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
        logging.info("Pushing to the hub...")
        train_testvalid["train"].push_to_hub(
            config["main"]["checkpoint"],
            config_name=config["ds"]["config"],
            split="train",
        )
        test_valid["test"].push_to_hub(
            config["main"]["checkpoint"],
            config_name=config["ds"]["config"],
            split="test",
        )
        test_valid["train"].push_to_hub(
            config["main"]["checkpoint"],
            config_name=config["ds"]["config"],
            split="valid",
        )
