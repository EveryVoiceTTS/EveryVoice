import re
from collections import Counter
from itertools import chain
from typing import Dict, List

from loguru import logger
from nltk.tokenize import RegexpTokenizer

from everyvoice.config.text_config import TextConfig
from everyvoice.exceptions import ConfigError
from everyvoice.text.features import get_features


class TextProcessor:
    def __init__(self, config: TextConfig):
        self.config = config
        self._all_symbols = self.config.symbols.model_dump()
        if "pad" in self._all_symbols:
            assert isinstance(self._all_symbols["pad"], str)
        self._pad_symbol = self._all_symbols["pad"]
        # apply longest characters first to apply multigraph symbols first
        self.symbols = sorted(
            list(
                chain.from_iterable(
                    list(v)
                    for k, v in self._all_symbols.items()
                    if k not in ("pad", "punctuation")
                )
            ),
            key=len,
            reverse=True,
        )
        self.symbols.insert(0, self._pad_symbol)
        # Add whitespace
        self.symbols.insert(1, " ")
        # Add punctuation
        # Add an internal hash to convert from the type of Punctuation to the internal representation
        self.punctuation_internal_hash = {
            "exclamations": "<EXCL>",
            "question_symbols": "<QINT>",
            "quotemarks": "<QUOTE>",
            "big_breaks": "<BB>",
            "small_breaks": "<SB>",
        }
        # Create a hash table from punctuation to the internal ID
        self.punctuation_to_internal_id = {}
        self.punctuation_characters = []
        for (
            punctuation_type,
            punctuation_type_values,
        ) in self.config.symbols.punctuation.model_dump().items():
            self.punctuation_characters += punctuation_type_values
            self.punctuation_to_internal_id.update(
                {
                    v: self.punctuation_internal_hash[punctuation_type]
                    for v in punctuation_type_values
                }
            )

        # Add the internal punctuation IDs to the symbols list
        self.symbols += list(self.punctuation_internal_hash.values())
        self.to_replace = config.to_replace
        self.missing_symbols: Counter[str] = Counter()
        self.duplicate_symbols: Counter[str] = Counter()

        # Mappings from symbol to numeric ID and vice versa
        self._symbol_to_id: Dict[str, int] = {}
        self._id_to_symbol: Dict[int, str] = {}
        for i, s in enumerate(self.symbols):
            if s in self._symbol_to_id:
                logger.warning(
                    f"Symbol '{s}' has already been declared at position {self._symbol_to_id[s]} so we will use that index instead of the current index {i}. Please remove duplicates from your configuration."
                )
                self.duplicate_symbols[s] += 1
            else:
                self._symbol_to_id[s] = i
                self._id_to_symbol[i] = s

        self._tokenizer = RegexpTokenizer(
            "|".join([re.escape(x) for x in self.symbols + self.punctuation_characters])
        )
        self._missing_symbol_finder = RegexpTokenizer(
            "|".join(
                [re.escape(x) for x in self.symbols + self.punctuation_characters]
            ),
            gaps=True,
            discard_empty=True,
        )

    def punctuation_cleaner(self, tokenized_text: list[str]) -> list[str]:
        """Given some text, normalize all punctuation according to internal representation

        Args:
            text (list[str]): tokenized text with punctuation

        Returns:
            list[str]: tokenized text with normalized punctuation
        """
        return [
            self.punctuation_to_internal_id.get(token, token)
            for token in tokenized_text
        ]

    def replace_cleaner(self, text: str) -> str:
        """Given some text and a list of replacement operations in the form of input/output key value pairs,
           return the transformed text.
        Args:
            text (str): The text to be converted
        Returns:
            str: the replaced text
        """
        for k, v in self.to_replace.items():
            text = re.sub(k, v, text)
        return text

    def text_to_sequence(self, text: str, quiet: bool = False):
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        cleaner_fns: a list of fns to clean text
        Returns:
        List of integers corresponding to the symbols in the text
        """
        sequence = []
        clean_tokens = self.text_to_tokens(text, quiet)
        for symbol in clean_tokens:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence

    def text_to_phonological_features(self, text: str, quiet: bool = False):
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
            text: string to convert to a sequence
            quiet: suppress warnings
        Returns:
            List of phonological feature vectors
        """
        clean_text = self.text_to_tokens(text, quiet)
        return get_features(clean_text)

    def clean_text(self, text: str) -> str:
        """Converts some text to cleaned text"""
        text = self.replace_cleaner(text)
        for cleaner_fn in self.config.cleaners:
            try:
                text = cleaner_fn(text)
            except Exception as e:
                raise ConfigError(
                    f"Cleaner did not work and threw exception {e}"
                ) from e
        return text

    def text_to_tokens(self, text: str, quiet: bool = False):
        """Converts a string of text to a sequence of tokens.
        Args:
            text: string to convert to a sequence
            quiet: suppress warnings
        Returns:
            List of symbols in the text
        """
        clean_text = self.clean_text(text)
        clean_tokens = self._tokenizer.tokenize(clean_text)
        for symbol in self._missing_symbol_finder.tokenize(clean_text):
            if not quiet:
                logger.warning(
                    f"Symbol '{symbol}' occurs in the text '{clean_text}' but was not declared in your configuration so it is being ignored."
                )
            self.missing_symbols[symbol] += 1
        return clean_tokens

    def get_missing_symbols(self, text):
        """Helper function to return a list of symbols missing from configuration."""
        clean_text = self.clean_text(text)
        return self._missing_symbol_finder.tokenize(clean_text)

    def cleaned_text_to_sequence(self, cleaned_text):
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
            text: string to convert to a sequence
        Returns:
            List of integers corresponding to the symbols in the text
        """
        cleaned_text = self._tokenizer.tokenize(cleaned_text)
        return [self._symbol_to_id[symbol] for symbol in cleaned_text]

    def token_sequence_to_text_sequence(self, sequence) -> List[str]:
        """Converts a sequence of IDs to a sequence of text characters"""
        return [self._id_to_symbol[symbol_id] for symbol_id in sequence]

    def token_sequence_to_text(self, sequence) -> str:
        """Converts a sequence of IDs back to a string"""
        return "".join(self.token_sequence_to_text_sequence(sequence))
