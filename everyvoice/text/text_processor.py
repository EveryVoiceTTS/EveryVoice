import re
from collections import Counter
from typing import Optional, Type, overload

import numpy as np
import numpy.typing as npt
from loguru import logger
from nltk.tokenize import RegexpTokenizer

from everyvoice.config.text_config import TextConfig
from everyvoice.exceptions import OutOfVocabularySymbolError
from everyvoice.text.features import PhonologicalFeatureCalculator
from everyvoice.text.phonemizer import AVAILABLE_G2P_ENGINES, get_g2p_engine
from everyvoice.text.utils import (
    apply_cleaners_helper,
    apply_to_replace_helper,
    normalize_text_helper,
)

PAD_SYMBOL = "\x80"


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
        self._pad_symbol = PAD_SYMBOL  # Use the Unicode PAD symbol

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
        self.punctuation_to_internal_id = {
            v: self.punctuation_internal_hash[punctuation_type]
            for punctuation_type, punctuation_type_values in iter(
                self.config.symbols.punctuation
            )
            for v in punctuation_type_values
        }
        self.punctuation_characters = list(self.punctuation_to_internal_id.keys())
        assert set(self.punctuation_characters) == self.config.symbols.punctuation.all

        # Add the internal punctuation IDs to the symbols list
        # Combine all the symbol fields into one list (except for punctuation)
        symbols = self.config.symbols.all_except_punctuation
        symbols |= set(self.punctuation_internal_hash.values())
        symbols |= self.config.symbols.punctuation.all
        # TODO: do I need to clean the symbols? How to do this if datasets have
        #       their own cleaners?
        _hardcoded_internal_symbols = [self._pad_symbol, " "]
        self.symbols = _hardcoded_internal_symbols + list(
            sorted(
                # Remove duplicates from symbol list, and apply longest
                # characters first to apply multigraph symbols first
                symbols - set(_hardcoded_internal_symbols),
                key=lambda symbol: (
                    -len(symbol),
                    symbol,
                ),  # reverse-length sort, then sort alphabetically
            )
        )
        self.to_replace = config.to_replace
        self.missing_symbols: Counter[str] = Counter()

        # Mappings from symbol to numeric ID and vice versa
        # dicts are preferred here to lists although list[index]
        # and list.index(symbol) could provide the same result
        # dicts are redundant protection against duplicate symbols
        self._symbol_to_id: dict[str, int] = {}
        self._id_to_symbol: dict[int, str] = {}
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
        return apply_to_replace_helper(text, self.to_replace)

    def apply_cleaners(self, text: str) -> str:
        """Converts some text to cleaned text

        Args:
            text (str): The text to be converted
        Returns:
            str: the replaced text

        >>> from everyvoice.utils import collapse_whitespace, lower, nfc_normalize
        >>> tp = TextProcessor(TextConfig(cleaners=[collapse_whitespace, lower, nfc_normalize]))
        >>> tp.apply_cleaners('HELLO\u0301')
        'helló'
        """
        return apply_cleaners_helper(text, self.config.cleaners)

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

        >>> from everyvoice.utils import collapse_whitespace, lower, nfc_normalize
        >>> tp = TextProcessor(TextConfig(cleaners=[collapse_whitespace, lower, nfc_normalize]))
        >>> tp.normalize_text('HELLO\u0301!')
        'helló!'
        """
        return normalize_text_helper(
            text,
            self.to_replace,
            self.config.cleaners,
            apply_cleaners=apply_cleaners,
            apply_replace_rules=apply_replace_rules,
        )

    def calculate_phonological_features(
        self, phone_tokens: list[str], apply_punctuation_rules=True
    ) -> npt.NDArray[np.float32]:
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
            text (list[str]): a list of IPA and normalized punctuation tokens
            apply_punctuation_rules (bool): whether to convert punctuation into discrete values like <BB> or <EXCL>. Defaults to True. If set to False, features.py will have to be amended to handle punctuation.
        Returns:
            npt.NDArray[np.float32]: a list of multi-hot phonological feature vectors

        >>> tp = TextProcessor(TextConfig())
        >>> tp.calculate_phonological_features(['aɪ'])
        array([[ 1.,  1., -1.,  1., -1., -1., -1.,  0.,  1., -1., -1.,  0., -1.,
                 0., -1.,  0.,  0., -1., -1., -1.,  0., -1.,  0.,  0.,  0.,  0.,
                 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],
              dtype=float32)
        """
        if self.phonological_feature_calculator is None:
            self.phonological_feature_calculator = PhonologicalFeatureCalculator(
                text_config=self.config,
                punctuation_hash=self.punctuation_internal_hash,
            )
        if apply_punctuation_rules:
            phone_tokens = self.apply_punctuation_rules(phone_tokens)
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
        punctuation_set = set(self.punctuation_characters)
        for token in tokens:
            if token in self._symbol_to_id or token in punctuation_set:
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
        normalize_punctuation: bool = False,
        encode_as_phonological_features: bool = False,
        lang_id: Optional[str] = None,
        quiet: bool = False,
        find_missing: bool = True,
    ) -> list[int] | npt.NDArray[np.float32]:
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
            text: string to convert to a sequence
        Returns:
            list[int]|list[list[int]]: Either a one-hot encoding of integers
                corresponding to the symbols in the text, or a multi-hot
                phonological feature vector

        >>> from everyvoice.config.text_config import Symbols
        >>> tp = TextProcessor(TextConfig(symbols=Symbols(ipa=['a', 'h', 'ʌ', 'l', 'o', 'ʊ'])))
        >>> tp.encode_text('hello \x80\x80', quiet=True) # e is not in the default symbols so it is ignored
        [19, 20, 20, 21, 1, 0, 0]
        >>> tp.encode_text('hello \x80\x80', apply_g2p=True, lang_id='boop', quiet=True)
        Traceback (most recent call last):
        ...
        ValueError: You tried to apply g2p for language 'boop', but no g2p engine exists for that language. Please see the <TODO: docs>.
        >>> tp.encode_text('hello \x80\x80', apply_g2p=False, lang_id='boop', encode_as_phonological_features=True, quiet=True)
        Traceback (most recent call last):
        ...
        ValueError: 'encode_as_phonological_features' was set to True but 'apply_g2p' was set to False. In order to calculate phonological features, you must first apply g2p to the text. Please set 'apply_g2p' to True.
        >>> tp.encode_text('hello \x80\x80', apply_g2p=True, lang_id='eng', quiet=True)
        [19, 27, 20, 21, 26, 1, 0, 0]
        """
        # Error states
        if encode_as_phonological_features and not apply_g2p:
            raise ValueError(
                "'encode_as_phonological_features' was set to True but 'apply_g2p' was set to False."
                " In order to calculate phonological features, you must first apply g2p to the text."
                " Please set 'apply_g2p' to True."
            )
        if apply_g2p and (lang_id is None or lang_id not in AVAILABLE_G2P_ENGINES):
            raise ValueError(
                f"You tried to apply g2p for language '{lang_id}', but no g2p engine exists for that language."
                " Please see the <TODO: docs>."
            )

        if normalize_text:
            text = self.normalize_text(text)
        if apply_g2p and lang_id is not None:
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
            # applying punctuation rules will have already happened here, so set to False
            return self.calculate_phonological_features(
                tokens, apply_punctuation_rules=False
            )
        else:
            # TODO: catch errors
            return [self._symbol_to_id[symbol] for symbol in tokens]

    def token_sequence_to_text_sequence(self, sequence: list[int]) -> list[str]:
        """Converts a sequence of IDs to a sequence of text characters

        Args:
            sequence (list[int]): a sequence of IDs

        Returns:
            list[str]: a sequence of text characters

        >>> tp = TextProcessor(TextConfig())
        >>> tp.token_sequence_to_text_sequence([0, 6, 0, 0])
        ['\x80', '<SIL>', '\x80', '\x80']
        """
        return [self._id_to_symbol[symbol_id] for symbol_id in sequence]

    def encode_string_tokens(self, sequence: list[str]) -> list[int]:
        """Encode a sequence of string tokens

        Args:
            sequence (list[str]): a list of string tokens

        Returns:
            list[int]: a list of token indices

        >>> tp = TextProcessor(TextConfig())
        >>> tp.encode_string_tokens(['\x80', '<SIL>', '\x80', '\x80'])
        [0, 6, 0, 0]
        """
        # TODO: catch errors
        encoded_tokens = []
        for string_token in sequence:
            try:
                encoded_tokens.append(self._symbol_to_id[string_token])
            except KeyError as e:
                raise OutOfVocabularySymbolError(
                    f"Sequence {sequence} contains item {string_token}"
                ) from e
        return encoded_tokens

    def encode_escaped_string_sequence(
        self, string_of_tokens: str, split_character="/"
    ):
        assert (
            len(split_character) >= 1
        ), "An escaped string sequence must have a character to split on (default is '/')"
        return self.encode_string_tokens(
            [token for token in string_of_tokens.split(split_character) if token]
        )

    @overload
    def decode_tokens(  # noqa E704
        self, sequence: list[int], join_character: Type[None]
    ) -> list[str]: ...

    @overload
    def decode_tokens(  # noqa E704
        self, sequence: list[int], join_character: str
    ) -> str: ...

    def decode_tokens(self, sequence: list[int], join_character="/") -> str | list[str]:
        """Decode a sequence of encoded phone or character tokens into a sequence of strings

        Args:
            sequence (List[int]): sequence of phone or character tokens

        Returns:
            str: the string equivalent of the sequence

        >>> tp = TextProcessor(TextConfig())
        >>> tp.decode_tokens([0, 1, 2, 0, 0])
        '\x80/ /<QUOTE>/\x80/\x80'
        """
        if join_character is None:
            return self.token_sequence_to_text_sequence(sequence)
        else:
            return join_character.join(self.token_sequence_to_text_sequence(sequence))
