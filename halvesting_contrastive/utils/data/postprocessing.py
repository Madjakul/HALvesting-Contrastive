# halvesting_geometric/utils/data/postprocessing.py

import math
import re
from collections import Counter
from typing import Any, Dict

import nltk
from langdetect import detect


class Postprocessing:
    """Clean up the sentence pairs."""

    @classmethod
    def run(cls, example: Dict[str, Any]):
        """Postprocess the data after sampling.

        Parameters
        ----------
        example: Dict[str, Any]
            The example to postprocess.

        Returns
        -------
        bool
            If the example is valid or not.
        """
        q, k = example["query_text"], example["key_text"]

        # 1) Enough words
        if not (cls.has_enough_words(q) and cls.has_enough_words(k)):
            return False

        # 2) English check
        # if not (cls.is_english(q) and cls.is_english(k)):
        #     return False

        # 3) Repetitiveness
        if cls.is_repetitive(q) or cls.is_repetitive(k):
            return False

        # 4) Too many symbols
        if cls.has_too_many_symbols(q) or cls.has_too_many_symbols(k):
            return False

        if "\t" in q or "\t" in k:
            return False

        return True

    @staticmethod
    def is_english(text: str):
        """Check if the text is in English.

        Parameters
        ----------
        text: str
            The text to check.

        Returns
        -------
        bool
            If the text is in English or not.
        """
        try:
            return detect(text) == "en"
        except:
            return False

    @staticmethod
    def has_enough_words(text: str):
        """Check if the text has more than 6 words.

        Parameters
        ----------
        text: str
            The text to check.

        Returns
        -------
        bool
            If the text has more than 6 words or not.
        """
        return len(nltk.word_tokenize(text)) > 6

    @staticmethod
    def has_too_many_symbols(text: str):
        """Check if the text has more than 25% symbols.

        Parameters
        ----------
        text: str
            The text to check.

        Returns
        -------
        bool
            If the text has more than 25% symbols or not.
        """
        symbols = re.findall(r"[^A-zÀ-ú]", text)
        return len(symbols) / len(text) > 0.25

    @staticmethod
    def is_repetitive(text: str):
        """Check if the text is repetitive. This is done by checking the
        entropy of the unigram distribution.

        Parameters
        ----------
        text: str
            The text to check.

        Returns
        -------
        bool
            If the text is repetitive or not
        """
        normalized_words = [word.lower() for word in nltk.word_tokenize(text)]
        counter = Counter(normalized_words)

        # calculate the entropy of the unigram distribution
        total = sum(counter.values())
        entropy = sum(
            map(
                lambda x: -x / total * math.log(x / total) if x > 0 else 0.0,
                counter.values(),
            )
        )

        return entropy < 1.5
