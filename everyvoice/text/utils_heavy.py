"""Symbol-mapping helpers built on panphon/scipy.
Particularly useful for mapping symbols to StyleTTS2 pre-trained text encoder.
"""

from typing import Callable, NamedTuple, Sequence

import numpy as np
from panphon.distance import Distance
from scipy.optimize import linear_sum_assignment

DistanceFn = Callable[[str, str], float]

_distance = Distance()


def unicode_distance(a: str, b: str) -> int:
    """A last-ditch effort to find a distance between non-IPA symbols."""
    assert len(a) == 1
    assert len(b) == 1
    return abs(ord(a) - ord(b))


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


def _is_recognized_ipa(symbol: str) -> bool:
    """Whether panphon can derive articulatory features for symbol at all."""
    return bool(_distance.fm.word_to_vector_list(symbol, numeric=True))


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
    return float(unicode_distance(a, b))


class SymbolMappingResult(NamedTuple):
    exact: list[str]
    suggestions: dict[str, str]
    distances: dict[str, float]
    unmapped: list[str]


def suggest_symbol_mapping(
    user_symbols: Sequence[str],
    pretrained_symbols: Sequence[str],
    distance_fn: DistanceFn = styletts2_symbol_distance,
) -> SymbolMappingResult:
    """Suggest how a user's declared symbols could be aligned onto a fixed pretrained symbol table.

    Symbols already present in `pretrained_symbols` are left untouched. Symbols
    that aren't are paired one-to-one with the closest pretrained symbols not
    already claimed by an exact match, so distinct user symbols never collapse
    onto the same pretrained symbol. If there are more novel symbols than free
    pretrained symbols to pair them with, the excess are reported as unmapped
    rather than given a suggestion.

    Args:
        user_symbols (Sequence[str]): the symbols declared in a user's TextConfig
        pretrained_symbols (Sequence[str]): the fixed symbol table of a pretrained model
        distance_fn (Callable[[str, str], float]): symbol-pair distance function

    Returns:
        SymbolMappingResult: exact matches, suggested substitutions, their
            distances, and any symbols that could not be mapped at all

    >>> result = suggest_symbol_mapping(['p', 'ʒ'], ['p', 'ʃ'])
    >>> result.exact
    ['p']
    >>> result.suggestions
    {'ʒ': 'ʃ'}
    """
    pretrained_set = set(pretrained_symbols)
    exact = [s for s in user_symbols if s in pretrained_set]
    oov = [s for s in user_symbols if s not in pretrained_set]
    available = [p for p in pretrained_symbols if p not in set(exact)]

    suggestions: dict[str, str] = {}
    distances: dict[str, float] = {}
    unmapped: list[str] = []
    if oov:
        mapping = find_optimal_mapping(oov, available, distance_fn)
        mapping_dict = dict(mapping)
        for symbol in oov:
            target = mapping_dict.get(symbol)
            if target is None:
                unmapped.append(symbol)
            else:
                suggestions[symbol] = target
                distances[symbol] = float(distance_fn(symbol, target))

    return SymbolMappingResult(
        exact=exact, suggestions=suggestions, distances=distances, unmapped=unmapped
    )
