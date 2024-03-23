import grapheme
from ipatok import tokenise


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
