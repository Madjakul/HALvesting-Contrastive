# halvesting_contrastive/utils/argparser/preprocess_argparse.py

import argparse


class PreprocessArgparse:
    """Argument parser used to preprocess HALvest."""

    @classmethod
    def parse_known_args(cls):
        """Parses arguments.

        Returns
        -------
        args: Any
            Parsed arguments.
        """
        parser = argparse.ArgumentParser(
            description="Argument parser used to preprocess HALvest's documents."
        )
        parser.add_argument(
            "--config_path",
            type=str,
            required=True,
            help="Path to the config file.",
        )
        parser.add_argument(
            "--do_convert_responses",
            action="store_true",
            help="Convert responses to metadata.",
        )
        parser.add_argument(
            "--responses_dir",
            type=str,
            default="./data/responses/",
            help="Path to the responses directory. Only used if do_convert_responses is True.",
        )
        parser.add_argument(
            "--metadata_path",
            type=str,
            default="./data/metadata.json",
            help="Path to the metadata file. If do_convert_responses is True, the metadata will be saved here.",
        )
        parser.add_argument(
            "--num_proc",
            type=int,
            default=None,
            help="Number of processes to use. Default is the number of CPUs.",
        )
        parser.add_argument(
            "--push_to_hub",
            action="store_true",
            help="Push the dataset to the Hub.",
        )
        args, _ = parser.parse_known_args()
        return args
