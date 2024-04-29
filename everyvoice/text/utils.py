import re

import grapheme
from ipatok import tokenise

from everyvoice.config.utils import PossiblySerializedCallable
from everyvoice.exceptions import ConfigError


def normalize_text_helper(
    text: str,
    to_replace: dict[str, str],
    cleaners: list[PossiblySerializedCallable],
    apply_replace_rules: bool = True,
    apply_cleaners: bool = True,
) -> str:
    """Helper for TextProcessor.normalize_text. We need to use it for the Symbols validator

    Args:
        text (str): The text to be normalized
        to_replace (dict[str, str]): replace rules to apply
        cleaners (list[PossiblySerializedCallable]): Cleaner functions to apply
        apply_replace_rules (bool): Whether to apply replace rules
        apply_cleaners (bool): Whether to apply cleaners

    Returns:
        str: the normalized text
    """
    if apply_replace_rules:
        text = apply_to_replace_helper(text, to_replace)
    if apply_cleaners:
        text = apply_cleaners_helper(text, cleaners)
    return text


def apply_to_replace_helper(text: str, to_replace: dict[str, str]) -> str:
    """Helper for TextProcessor.apply_replacement_rules. We need to use it for the Symbols validator

    Args:
        text (str): The text to be converted
        to_replace (dict[str, str]): replace rules to apply

    Returns:
        str: The text with replace rules applied
    """
    for k, v in to_replace.items():
        text = re.sub(k, v, text)
    return text


def apply_cleaners_helper(text: str, cleaners: list[PossiblySerializedCallable]) -> str:
    """Helper for TextProcesser.apply_cleaners. We need to use it for the Symbols validator

    Args:
        text (str): The text to be converted
        cleaners (list[PossiblySerializedCallable]): Cleaner functions to apply

    Raises:
        ConfigError: a cleaner threw an error

    Returns:
        str: cleaned text
    """
    for cleaner_fn in cleaners:
        try:
            text = cleaner_fn(text)
        except Exception as e:
            raise ConfigError(f"Cleaner did not work and threw exception {e}") from e
    return text


def guess_graphemes_in_text(text: str) -> set[str]:
    """Given some text, determine the set of graphemes by
        applying Unicode grapheme clustering rules.

    Args:
        text (str): some normalized, un-tokenized text
    Returns:
        set[str]: a set of possible graphemes

    >>> sorted(guess_graphemes_in_text('g\u0331an'))
    ['a', 'g̱', 'n']
    """
    return set(grapheme.graphemes(text))


def guess_graphemes_in_text_lines(text_lines: list[str]) -> set[str]:
    """Helper function for guessing graphemes in multiple lines of text,
        determined by applying Unicode grapheme clustering rules.

    Args:
        text (list[str]): some normalized, un-tokenized text separated line-by-line
    Returns:
        set[str]: a set of possible graphemes

    >>> example_data = [" කෝකටත් මං වෙනදා ", " ඇන්ජලීනා ජොලී කියන්නේ "]
    >>> sorted(guess_graphemes_in_text_lines(example_data))
    [' ', 'ඇ', 'ක', 'කි', 'කෝ', 'ජ', 'ජො', 'ට', 'ත්', 'දා', 'න', 'න්', 'නා', 'නේ', 'මං', 'ය', 'ලී', 'වෙ']
    """
    graphemes = set()
    for line in text_lines:
        graphemes.update(guess_graphemes_in_text(line))
    return graphemes


def guess_ipa_phones_in_text(text: str) -> set[str]:
    """Given some text, determine the set of valid ipa phones
        by applying strict IPA tokenization and discarding the rest

    Args:
        text (str): some normalized, un-tokenized text that has IPA characters
    Returns:
        set[str]: a set of possible IPA phones

    >>> sorted(guess_ipa_phones_in_text('ʃin1[}!]'))
    ['i', 'n', 'ʃ']

    """
    return set(tokenise(text, replace=False, tones=True, strict=False, unknown=False))


def guess_ipa_phones_in_text_lines(text_lines: list[str]) -> set[str]:
    """Given some text, determine the set of valid ipa phones
        by applying strict IPA tokenization and discarding the rest

    Args:
        text (list[str]): some normalized, un-tokenized text that has IPA characters separated line-by-line
    Returns:
        set[str]: a set of possible IPA phones

    # TODO: panphon doesn't agree with ipatok about whether g is valid IPA

    >>> example_data = ["ʃin", "gotcha"]
    >>> sorted(guess_graphemes_in_text_lines(example_data))
    ['a', 'c', 'g', 'h', 'i', 'n', 'o', 't', 'ʃ']
    """
    phones = set()
    for line in text_lines:
        phones.update(guess_ipa_phones_in_text(line))
    return phones
