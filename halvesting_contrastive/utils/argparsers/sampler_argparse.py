# halvesting_contrastive/utils/argparsers/sampler_argparse.py

import argparse


class SamplerArgparse:
    """Argument parser used to sample sentences, paragraphs or inverse cloze
    pairs."""

    @classmethod
    def parse_known_args(cls):
        """Parses arguments.

        Returns
        -------
        args: Any
            Parsed arguments.
        """
        parser = argparse.ArgumentParser(
            description="Argument parser used to sample sentences, paragraphs or inverse cloze pairs."
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
        parser.add_argument(
            "--output_dir",
            type=str,
            default="./data",
            help="Path to the output directory.",
        )
        args, _ = parser.parse_known_args()
        return args
