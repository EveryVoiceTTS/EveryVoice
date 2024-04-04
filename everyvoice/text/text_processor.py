import re
from collections import Counter
from itertools import chain
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from loguru import logger
from nltk.tokenize import RegexpTokenizer

from everyvoice.config.text_config import TextConfig
from everyvoice.exceptions import ConfigError, OutOfVocabularySymbol
from everyvoice.text.features import PhonologicalFeatureCalculator
from everyvoice.text.phonemizer import AVAILABLE_G2P_ENGINES, get_g2p_engine


class TextProcessor:
    """Text is processed like:

    InputText (str, either DatasetTextRepresentation.characters, DatasetTextRepresentation.ipa_phones, or DatasetTextRepresentation.arpabet)
      -> to_replace operations
        -> cleaner operations
        = Cleaned Text (str)
          -> Optional[grapheme-to-phoneme, outputs tokens] OR
          -> tokenization
          = Tokens (list[str])
            -> Punctuation mapped to internal representation
            = Tokens with internal punctuation representation (list[str])
    """

    def __init__(self, config: TextConfig):
        self.config = config
        self.phonological_feature_calculator: Optional[
            PhonologicalFeatureCalculator
        ] = None
        self._all_symbols = self.config.symbols.model_dump()
        self._pad_symbol = "\x80"  # Use the Unicode PAD symbol
        # Combine all the symbol fields into one list (except for punctuation)
        self.symbols = list(
            chain.from_iterable(
                list(v) for k, v in self._all_symbols.items() if k != "punctuation"
            )
        )
        # Keep a list of valid punctuation
        self.punctuation = set(
            item
            for cat in self.config.symbols.punctuation.model_dump().values()
            for item in cat
        )
        # Add punctuation
        # Add an internal hash to convert from the type of Punctuation to the internal representation
        self.punctuation_internal_hash = {
            "exclamations": "<EXCL>",
            "question_symbols": "<QINT>",
            "quotemarks": "<QUOTE>",
            "big_breaks": "<BB>",
            "small_breaks": "<SB>",
            "ellipsis": "<EPS>",
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
        # TODO: do I need to clean the symbols? How to do this if datasets have their own cleaners
        # Remove duplicates from symbol list, and apply longest characters first
        # to apply multigraph symbols first
        self._hardcoded_internal_symbols = [self._pad_symbol, " "]
        self.symbols = self._hardcoded_internal_symbols + [
            x
            for x in sorted(
                set(self.symbols),
                key=lambda symbol: (
                    -len(symbol),
                    symbol,
                ),  # reverse-length sort, then sort alphabetically
            )
            if x not in self._hardcoded_internal_symbols
        ]
        self.to_replace = config.to_replace
        self.missing_symbols: Counter[str] = Counter()

        # Mappings from symbol to numeric ID and vice versa
        self._symbol_to_id: Dict[str, int] = {}
        self._id_to_symbol: Dict[int, str] = {}
        for i, s in enumerate(self.symbols):
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

    def get_missing_symbols(
        self, text: str, normalize_text=True, quiet=False
    ) -> list[str]:
        """Helper function to return a list of symbols missing from configuration.

        Args:
            text (str): text to find missing symbols in
            normalize_text (bool, optional): whether to normalize text first. Defaults to True.

        Returns:
            list[str]: a list of missing symbols in the text. globs all adjacent missing symbols together

        >>> tp = TextProcessor(TextConfig())
        >>> tp.get_missing_symbols(' ç **', quiet=True)
        ['ç', '**']
        """
        if normalize_text:
            text = self.normalize_text(text)
        for symbol in (missing_tokens := self._missing_symbol_finder.tokenize(text)):
            if not quiet:
                logger.warning(
                    f"Symbol '{symbol}' occurs in the text '{text}' but was not declared in your configuration so it is being ignored."
                )
            self.missing_symbols[symbol] += 1
        return missing_tokens

    def apply_punctuation_rules(self, tokenized_text: list[str]) -> list[str]:
        """Given some text, normalize all punctuation according to internal representation

        Args:
            text (list[str]): tokenized text with punctuation

        Returns:
            list[str]: tokenized text with normalized punctuation

        >>> tp = TextProcessor(TextConfig())
        >>> tp.apply_punctuation_rules(['h', 'e', 'l', 'l', 'o', '.'])
        ['h', 'e', 'l', 'l', 'o', '<BB>']

        """
        return [
            self.punctuation_to_internal_id.get(token, token)
            for token in tokenized_text
        ]

    def apply_replacement_rules(self, text: str) -> str:
        """Given some text and a list of replacement operations in the form of input/output key value pairs,
           return the transformed text.
        Args:
            text (str): The text to be converted
        Returns:
            str: the replaced text

        >>> tp = TextProcessor(TextConfig(to_replace={'a': 'b'}))
        >>> tp.apply_replacement_rules('a')
        'b'

        """
        for k, v in self.to_replace.items():
            text = re.sub(k, v, text)
        return text

    def apply_cleaners(self, text: str) -> str:
        """Converts some text to cleaned text

        Args:
            text (str): The text to be converted
        Returns:
            str: the replaced text

        >>> tp = TextProcessor(TextConfig())
        >>> tp.apply_cleaners('HELLO\u0301')
        'helló'

        """
        for cleaner_fn in self.config.cleaners:
            try:
                text = cleaner_fn(text)
            except Exception as e:
                raise ConfigError(
                    f"Cleaner did not work and threw exception {e}"
                ) from e
        return text

    def normalize_text(
        self, text: str, apply_replace_rules=True, apply_cleaners=True
    ) -> str:
        """Normalize text by applying replace rules and all defined cleaners

        Args:
            text (str): un-normalized text
            apply_replace_rules (bool, optional): Whether to apply replace rules. Defaults to True.
            apply_cleaners (bool, optional): Whether to apply cleaners. Defaults to True.

        Returns:
            str: normalized text ready to be tokenized

        >>> tp = TextProcessor(TextConfig())
        >>> tp.normalize_text('HELLO\u0301!')
        'helló!'

        """
        if apply_replace_rules:
            text = self.apply_replacement_rules(text)
        if apply_cleaners:
            text = self.apply_cleaners(text)
        return text

    def calculate_phonological_features(
        self, phone_tokens: list[str]
    ) -> npt.NDArray[np.int_]:
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
            text (list[str]): a list of IPA and normalized punctuation tokens
        Returns:
            npt.NDArray[np.int_]: a list of multi-hot phonological feature vectors

        >>> tp = TextProcessor(TextConfig())
        """
        if self.phonological_feature_calculator is None:
            self.phonological_feature_calculator = PhonologicalFeatureCalculator(
                text_config=self.config, punctuation_hash=self.punctuation_internal_hash
            )
        return self.phonological_feature_calculator.get_features(phone_tokens)

    def apply_g2p_and_tokenization(
        self, normalized_text: str, lang_id: str, find_missing=True, quiet=False
    ) -> list[str]:
        """Converts a normalized string of graphemes for a particular language into a list of phone tokens.

        Args:
            normalized_text (str): a normalized string of graphemes
            lang_id (str): the language id

        Returns:
            list[str]: a list of phone tokens

        >>> from everyvoice.config.text_config import Symbols
        >>> tp = TextProcessor(TextConfig(symbols=Symbols(ipa=['a', 'h', 'ʌ', 'l', 'o', 'ʊ'])))
        >>> tp.apply_g2p_and_tokenization('hello', 'eng')
        ['h', 'ʌ', 'l', 'o', 'ʊ']

        """
        g2p_engine = get_g2p_engine(lang_id)
        try:
            tokens = g2p_engine(normalized_text)
        except Exception:
            tokens = None
            # TODO: do something here
            pass
        assert isinstance(
            tokens, list
        ), f"The g2p engine for {lang_id} produced {type(tokens)} but must produce a list of tokenized phones."
        valid_tokens = []
        for token in tokens:
            if token in self._symbol_to_id or token in self.punctuation:
                valid_tokens.append(token)
            else:
                if find_missing:
                    if not quiet:
                        logger.warning(
                            f"Symbol '{token}' occurs in the text '{normalized_text}' but was not declared in your configuration so it is being ignored."
                        )
                self.missing_symbols[token] += 1
                continue

        return valid_tokens

    def apply_tokenization(
        self, normalized_text: str, quiet: bool = False, find_missing=True
    ) -> list[str]:
        """Converts a string of normalized text to a sequence of tokens.

        Args:
            text (str): string to convert to a sequence
            quiet (bool): suppress warnings
            find_missing (bool): find missing tokens and log them as part of the TextProcessor.missing_symbols counter
        Returns:
            list[str]: List of symbols in the text without missing symbols

        >>> tp = TextProcessor(TextConfig())
        >>> tp.apply_tokenization('\x80\x80 *', quiet=True)
        ['\x80', '\x80', ' ']
        >>> tp.missing_symbols['*']
        1

        """
        if find_missing:
            self.get_missing_symbols(normalized_text, quiet=quiet)
        tokens = self._tokenizer.tokenize(normalized_text)
        return tokens

    def encode_text(
        self,
        text: str,
        normalize_text: bool = True,
        apply_g2p: bool = False,
        normalize_punctuation: bool = True,
        encode_as_phonological_features: bool = False,
        lang_id: Optional[str] = None,
        quiet: bool = False,
        find_missing: bool = True,
    ) -> list[int] | npt.NDArray[np.int_]:
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
            text: string to convert to a sequence
        Returns:
            list[int]|list[list[int]]: Either a one-hot encoding of integers corresponding to the symbols in the text, or a multi-hot phonological feature vector

            >>> from everyvoice.config.text_config import Symbols
            >>> tp = TextProcessor(TextConfig(symbols=Symbols(ipa=['a', 'h', 'ʌ', 'l', 'o', 'ʊ'])))
            >>> tp.encode_text('hello \x80\x80', quiet=True) # e is not in the default symbols so it is ignored
            [4, 6, 6, 7, 1, 0, 0]
            >>> tp.encode_text('hello \x80\x80', apply_g2p=True, lang_id='boop', quiet=True)
            Traceback (most recent call last):
            ...
            ValueError: You tried to apply g2p for language 'boop', but no g2p engine exists for that language. Please see the <TODO: docs>.
            >>> tp.encode_text('hello \x80\x80', apply_g2p=False, lang_id='boop', encode_as_phonological_features=True, quiet=True)
            Traceback (most recent call last):
            ...
            ValueError: 'encode_as_phonological_features' was set to True but 'apply_g2p' was set to False. In order to calculate phonological features, you must first apply g2p to the text. Please set 'apply_g2p' to True.
            >>> tp.encode_text('hello \x80\x80', apply_g2p=True, lang_id='eng', quiet=True)
            [4, 5, 6, 7, 8, 1, 0, 0]

        """
        # Error states
        if encode_as_phonological_features and not apply_g2p:
            raise ValueError(
                "'encode_as_phonological_features' was set to True but 'apply_g2p' was set to False. In order to calculate phonological features, you must first apply g2p to the text. Please set 'apply_g2p' to True."
            )
        if apply_g2p and (lang_id is None or lang_id not in AVAILABLE_G2P_ENGINES):
            raise ValueError(
                f"You tried to apply g2p for language '{lang_id}', but no g2p engine exists for that language. Please see the <TODO: docs>."
            )

        if normalize_text:
            text = self.normalize_text(text)
        if apply_g2p:
            tokens = self.apply_g2p_and_tokenization(
                normalized_text=text,
                lang_id=lang_id,
                quiet=quiet,
                find_missing=find_missing,
            )
        else:
            tokens = self.apply_tokenization(
                text, quiet=quiet, find_missing=find_missing
            )
        if normalize_punctuation:
            tokens = self.apply_punctuation_rules(tokens)
        if encode_as_phonological_features:
            return self.calculate_phonological_features(tokens).tolist()
        else:
            # TODO: catch errors
            return [self._symbol_to_id[symbol] for symbol in tokens]

    def _token_sequence_to_text_sequence(self, sequence) -> List[str]:
        """Converts a sequence of IDs to a sequence of text characters"""
        return [self._id_to_symbol[symbol_id] for symbol_id in sequence]

    def encode_string_tokens(self, sequence: list[str]) -> list[int]:
        """Encode a sequence of string tokens

        Args:
            sequence (list[str]): a list of string tokens

        Returns:
            list[int]: a list of token indices

        >>> tp = TextProcessor(TextConfig())
        >>> tp.decode_tokens(['\x80', '<SIL>', '\x80', '\x80'])
        [0, 1, 2, 0, 0]
        """
        # TODO: catch errors
        encoded_tokens = []
        for string_token in sequence:
            try:
                encoded_tokens.append(self._symbol_to_id[string_token])
            except KeyError as e:
                raise OutOfVocabularySymbol(
                    f"Sequence {sequence} contains item {string_token}"
                ) from e
        return encoded_tokens

    def encode_escaped_string_sequence(
        self, string_of_tokens: str, split_character="/"
    ):
        assert (
            len(split_character) >= 1
        ), "An escaped string sequence must have a character to split on (default is '/')"
        return self.encode_string_tokens(string_of_tokens.split(split_character))

    def decode_tokens(self, sequence: List[int], join_character="/") -> str:
        """Decode a sequence of encoded phone or character tokens into a sequence of strings

        Args:
            sequence (List[int]): sequence of phone or character tokens

        Returns:
            str: the string equivalent of the sequence

        >>> tp = TextProcessor(TextConfig())
        >>> tp.decode_tokens([0, 1, 2, 0, 0])
        '\x80 <SIL>\x80\x80'

        """
        return join_character.join(self._token_sequence_to_text_sequence(sequence))
