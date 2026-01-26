# halvesting_geometric/utils/data/postprocessing.py

import math
import re
import statistics
from collections import Counter
from typing import Any, Dict

import nltk
from nltk.corpus import stopwords

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class Postprocessing:
    """Clean up the sentence triplets."""

    HEADER_TERMS_RE = re.compile(
        r"\b(keyword|keywords|a b s t r a c t)\b",
        re.IGNORECASE,
    )
    MATH_SYMBOLS_RE = re.compile(r"[\*∀≡→∧√ΣΠθωμλ_^\<\>\|≥≤±≈≠]")

    @classmethod
    def run(cls, example: Dict[str, Any]):
        """Postprocess the data after sampling."""
        q, p, n = example["query"], example["positive"], example["negative"]

        for text in [q, p, n]:
            words = nltk.word_tokenize(text)
            if (
                not cls.has_enough_alpha_words(words)
                or cls.has_abnormal_layout(text)
                or cls.contains_header_terms(text)
                or cls.is_heavily_symbolic(text)
                or not cls.has_enough_stop_words(words)
                or cls.is_repetitive(words)
                or cls.has_too_many_symbols(text)
                or cls.is_formula_or_code(text, words)
                or cls.is_malformed_sentence(text)
                or cls.contains_url_or_markup(text)
                or cls.has_high_uppercase_ratio(words)
                or cls.has_high_sentence_length_variance(text)
                or "\t" in text
            ):
                return False

        return True

    @staticmethod
    def has_enough_alpha_words(words: list[str], min_words: int = 10):
        alpha_words = [word for word in words if word.isalpha()]
        return len(alpha_words) >= min_words

    @staticmethod
    def has_abnormal_layout(text: str, max_newline_ratio: int = 2):
        """Filters text with list-like structures (e.g., author lists)."""
        if "@" in text:  # Email addresses are a clear sign of an author block
            return True

        num_newlines = text.count("\n")
        if num_newlines < 2:
            return False

        sentences = nltk.sent_tokenize(text)
        num_sentences = len(sentences)

        if num_sentences == 0:
            return True

        # If newline count is much higher than sentence count, it's likely a list
        return (num_newlines / num_sentences) > max_newline_ratio

    @classmethod
    def contains_header_terms(cls, text: str):
        """Searches for keywords that indicate academic headers or
        abstracts."""
        return bool(cls.HEADER_TERMS_RE.search(text))

    @classmethod
    def is_heavily_symbolic(cls, text: str, max_symbol_ratio: float = 0.05):
        """Aggressively filters text with a high density of math symbols."""
        if not text:
            return True
        math_symbols = cls.MATH_SYMBOLS_RE.findall(text)
        return (len(math_symbols) / len(text)) > max_symbol_ratio

    @staticmethod
    def has_enough_stop_words(words: list[str], ratio: float = 0.05):
        if not words:
            return False
        stop_word_count = sum(1 for word in words if word.lower() in STOP_WORDS)
        return (stop_word_count / len(words)) >= ratio

    @staticmethod
    def is_repetitive(words: list[str], min_entropy: float = 1.8):
        if not words:
            return True
        normalized_words = [word.lower() for word in words]
        counter = Counter(normalized_words)
        total = sum(counter.values())
        entropy = -sum((c / total) * math.log(c / total) for c in counter.values())
        return entropy < min_entropy

    @staticmethod
    def has_too_many_symbols(text: str, max_ratio: float = 0.30):
        if not text:
            return True
        symbols = re.findall(r"[^A-zÀ-ú\s\d\.,'\"!?]", text)
        return (len(symbols) / len(text)) > max_ratio

    @staticmethod
    def is_formula_or_code(
        text: str,
        words: list[str],
        digit_letter_ratio: float = 0.15,
        single_char_word_ratio: float = 0.20,
    ):
        if not text:
            return True
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        if letters == 0:
            return True
        if (digits / letters) > digit_letter_ratio:
            return True
        if not words:
            return True
        single_char_words = sum(
            1 for word in words if len(word) == 1 and word.isalpha()
        )
        if (single_char_words / len(words)) > single_char_word_ratio:
            return True
        return False

    @staticmethod
    def is_malformed_sentence(text: str):
        stripped_text = text.strip()
        if not stripped_text:
            return True

        if not stripped_text[0].isupper() and not stripped_text[0].isdigit():
            return True

        lower_text = stripped_text.lower()

        if lower_text.count("(") != lower_text.count(")") or lower_text.count(
            "["
        ) != lower_text.count("]"):
            return True
        if not lower_text.endswith((".", "?", "!", '"', "'")):
            return True
        if lower_text.endswith(
            ("et al.", "e.g.", "i.e.", "a.d.", "b.c.", "fig.", "tab.")
        ):
            return True

        return False

    @staticmethod
    def contains_url_or_markup(text: str):
        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        html_pattern = re.compile(r"<.*?>")
        return bool(url_pattern.search(text) or html_pattern.search(text))

    @staticmethod
    def has_high_uppercase_ratio(words: list[str], max_ratio: float = 0.40):
        if len(words) < 5:
            return False
        uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        return (uppercase_words / len(words)) > max_ratio

    @staticmethod
    def has_high_sentence_length_variance(text: str, max_cv: float = 0.6):
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 1:
            return False
        sentence_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
        if not sentence_lengths or sum(sentence_lengths) == 0:
            return True
        mean_len = statistics.mean(sentence_lengths)
        if mean_len < 4:
            return False
        stdev_len = statistics.stdev(sentence_lengths)
        cv = stdev_len / mean_len
        return cv > max_cv
