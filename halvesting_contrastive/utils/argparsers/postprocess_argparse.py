# halvesting_contrastive/utils/argparsers/postprocess_argparse.py

import argparse


class PostprocessArgparse:
    """Argument parser used to postprocess HALvest Contrastive."""

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
            "--num_proc",
            type=int,
            default=None,
            help="Number of processes to use. Default is the number of CPUs.",
        )
        args, _ = parser.parse_known_args()
        return args
