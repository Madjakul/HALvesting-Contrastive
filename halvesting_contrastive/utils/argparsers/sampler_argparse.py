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
        args, _ = parser.parse_known_args()
        return args
