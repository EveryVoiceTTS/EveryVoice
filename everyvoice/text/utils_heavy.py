"""Symbol-mapping helpers built on panphon/scipy.
Particularly useful for mapping symbols to StyleTTS2 pre-trained text encoder.

Note: many of the mapping functions in this module are only intended to be used
as hopefully slightly-better-than-random-assignment heuristics. Users should ultimately
rely on their knowledge of the language they are modelling to choose how symbols should
be represented in their data

"""

import unicodedata
from functools import cache
from typing import Callable, NamedTuple, Sequence

import numpy as np
from panphon.distance import Distance
from scipy.optimize import linear_sum_assignment

DistanceFn = Callable[[str, str], float]

_distance = Distance()


@cache
def _symbol_base(symbol: str) -> str:
    """The base character after stripping combining marks via canonical
    decomposition, e.g. 'é' -> 'e'."""
    decomposed = unicodedata.normalize("NFD", symbol)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _codepoint_tiebreak(a: str, b: str) -> float:
    """Squash raw codepoint distance into [0, 1) so it can order symbols
    within a unicode_table_distance tier (e.g. keeping digit '1' closer to
    '2' than to '4') without its effectively unbounded range (codepoints
    span up to 0x10FFFF) letting it dominate the tier itself."""
    raw = abs(ord(a) - ord(b))
    return raw / (raw + 1000)


def unicode_table_distance(a: str, b: str) -> float:
    """A last-ditch distance between symbols panphon doesn't recognize as IPA,
    using properties from the Unicode character database.

    Symbols are compared tier by tier, most to least similar, with tiers
    spaced widely enough that the codepoint tiebreaker used to order symbols
    within a tier can never cross into the next one:
      0.0:      identical
      (0, 1):   share a canonical base letter, e.g. 'é' and 'ê' both -> 'e'
      [1, 2):   same Unicode general category, e.g. both decimal digits (Nd)
      [2, 3):   same general category major class, e.g. both letters (L*)
      [3, 4):   no shared property found

    >>> unicode_table_distance('é', 'ê') < unicode_table_distance('é', 'ç')
    True
    """
    assert len(a) == 1
    assert len(b) == 1
    if a == b:
        return 0.0
    tiebreak = _codepoint_tiebreak(a, b)
    if _symbol_base(a) == _symbol_base(b):
        return tiebreak
    category_a = unicodedata.category(a)
    category_b = unicodedata.category(b)
    if category_a == category_b:
        return 1.0 + tiebreak
    if category_a[0] == category_b[0]:
        return 2.0 + tiebreak
    return 3.0 + tiebreak


def find_optimal_mapping(
    symbol_set_a: Sequence[str],
    symbol_set_b: Sequence[str],
    distance_fn: DistanceFn,
) -> list[tuple[str, str]]:
    """Find the one-to-one pairing between two symbol sets that minimizes total distance.

    Args:
        symbol_set_a (Sequence[str]): symbols to map from
        symbol_set_b (Sequence[str]): symbols to map to
        distance_fn (Callable[[str, str], float]): scores how dissimilar two
            symbols are; lower means more similar.

    Returns:
        list[tuple[str, str]]: the matched (a, b) pairs

    >>> mapping = find_optimal_mapping(['1', '8'], ['0', '9'], lambda a, b: abs(int(a) - int(b)))
    >>> sorted(mapping)
    [('1', '0'), ('8', '9')]
    """
    # empty set optimization
    if not symbol_set_a or not symbol_set_b:
        return []
    distance_matrix = np.array(
        [[distance_fn(a, b) for b in symbol_set_b] for a in symbol_set_a]
    )
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    mapping = [(symbol_set_a[r], symbol_set_b[c]) for r, c in zip(row_ind, col_ind)]
    return mapping


@cache
def _is_recognized_ipa(symbol: str) -> bool:
    """Whether panphon can derive articulatory features for symbol at all.

    Cached because find_optimal_mapping's distance matrix calls this once per
    (a, b) pair via styletts2_symbol_distance, re-checking the same symbol up
    to len(symbol_set) times over rather than once.
    """
    return bool(_distance.fm.word_to_vector_list(symbol, numeric=True))


def is_recognized_ipa_or_latin(symbol: str) -> bool:
    """Whether symbol is either a recognized IPA segment or made up entirely of
    Latin-script characters -- the two categories StyleTTS2's pretrained symbol
    table is mostly built from (basic a-z plus a handful of IPA letters), so mapping suggestions for such symbols tend to be meaningful.

    Symbols in neither category (e.g. Sinhala, Devanagari, or Thai graphemes)
    fall back to unicode_table_distance's generic Unicode-property comparison,
    which produces a low-confidence, largely arbitrary suggestion -- callers
    should warn that such symbols are better romanized or phonemized first.

    >>> is_recognized_ipa_or_latin('p')
    True
    >>> is_recognized_ipa_or_latin('é')
    True
    >>> is_recognized_ipa_or_latin('ක')
    False
    """
    return _is_recognized_ipa(symbol) or all(
        unicodedata.name(c, "").startswith("LATIN") for c in symbol
    )


def styletts2_symbol_distance(a: str, b: str) -> float:
    """Distance between two symbols for mapping onto a pretrained symbol table.

    Uses panphon's articulatory-feature-weighted edit distance when both symbols
    are recognized IPA segments (this also handles multi-character phones like
    diphthongs sensibly). Falls back to plain edit distance for multi-length comparisons
    and a simple Unicode table distance otherwise:
    panphon's feature vectors are all-zero for non-IPA symbols, which would
    otherwise make every grapheme equidistant from every other one.

    >>> styletts2_symbol_distance('p', 'p')
    0.0
    >>> styletts2_symbol_distance('p', 'b') < styletts2_symbol_distance('p', 'a')
    True
    """
    if _is_recognized_ipa(a) and _is_recognized_ipa(b):
        return float(_distance.weighted_feature_edit_distance(a, b))
    if len(a) > 1 or len(b) > 1:
        return float(_distance.fast_levenshtein_distance(a, b))
    return unicode_table_distance(a, b)


class SymbolMappingResult(NamedTuple):
    exact: list[str]
    suggestions: dict[str, str]
    distances: dict[str, float]
    unmapped: list[str]


# Punctuation-category characters that are nonetheless conventionally read
# aloud as words, e.g. '&' -> "and". Unicode doesn't separate these from
# purely typographic punctuation like '.' or ',', so this list is curated.
_PRONOUNCED_PUNCTUATION = {"&", "@", "#", "%"}


def _needs_text_normalization(symbol: str) -> bool:
    """Whether symbol is made up entirely of characters that are
    conventionally spelled out by text normalization rather than pronounced
    or transliterated as-is: digits, currency signs, math symbols (Unicode
    major classes N/Sc/Sm), or _PRONOUNCED_PUNCTUATION."""

    def _is_unpronounceable(c: str) -> bool:
        category = unicodedata.category(c)
        return (
            category[0] == "N"
            or category in ("Sc", "Sm")
            or c in _PRONOUNCED_PUNCTUATION
        )

    return len(symbol) > 0 and all(_is_unpronounceable(c) for c in symbol)


def suggest_symbol_mapping(
    user_symbols: Sequence[str],
    pretrained_symbols: Sequence[str],
    distance_fn: DistanceFn = styletts2_symbol_distance,
    reserved_targets: Sequence[str] = (),
) -> SymbolMappingResult:
    """Suggest a one-to-one mapping from user_symbols onto pretrained_symbols.

    Exact matches are left untouched; the rest are paired with the closest
    free pretrained symbol, excluding `reserved_targets`, space, '$', and
    symbols needing text normalization (see `_needs_text_normalization`).
    Symbols left with no free match are reported as unmapped.

    Args:
        user_symbols (Sequence[str]): the symbols declared in a user's TextConfig
        pretrained_symbols (Sequence[str]): the fixed symbol table of a pretrained model
        distance_fn (Callable[[str, str], float]): symbol-pair distance function
        reserved_targets (Sequence[str]): pretrained symbols to never suggest

    Returns:
        SymbolMappingResult: exact matches, suggestions, distances, unmapped

    >>> result = suggest_symbol_mapping(['p', 'ʒ'], ['p', 'ʃ'])
    >>> result.exact
    ['p']
    >>> result.suggestions
    {'ʒ': 'ʃ'}

    >>> result = suggest_symbol_mapping(['5', '&'], ['a'])
    >>> result.suggestions
    {}
    >>> sorted(result.unmapped)
    ['&', '5']
    """
    pretrained_set = set(pretrained_symbols)
    exact = [s for s in user_symbols if s in pretrained_set]
    oov = [s for s in user_symbols if s not in pretrained_set]
    needs_normalization = [s for s in oov if _needs_text_normalization(s)]
    mappable = [s for s in oov if not _needs_text_normalization(s)]
    # never suggest word-separator space or StyleTTS2's pad symbol as a target
    excluded_targets = set(exact) | set(reserved_targets) | {" ", "$"}
    available = [p for p in pretrained_symbols if p not in excluded_targets]

    suggestions: dict[str, str] = {}
    distances: dict[str, float] = {}
    unmapped: list[str] = list(needs_normalization)
    if mappable:
        mapping = find_optimal_mapping(mappable, available, distance_fn)
        mapping_dict = dict(mapping)
        for symbol in mappable:
            target = mapping_dict.get(symbol)
            if target is None:
                unmapped.append(symbol)
            else:
                suggestions[symbol] = target
                distances[symbol] = float(distance_fn(symbol, target))

    return SymbolMappingResult(
        exact=exact, suggestions=suggestions, distances=distances, unmapped=unmapped
    )
