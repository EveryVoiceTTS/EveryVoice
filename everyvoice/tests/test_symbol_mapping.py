#!/usr/bin/env python

import sys
from unittest import TestCase

from pytest import main

from everyvoice.text.utils_heavy import (
    find_optimal_mapping,
    styletts2_symbol_distance,
    suggest_symbol_mapping,
)


class SymbolMappingTest(TestCase):
    """Tests for the symbol-mapping utilities in everyvoice.text.utils_heavy"""

    def test_find_optimal_mapping_minimizes_total_distance(self):
        mapping = find_optimal_mapping(
            ["1", "8"], ["0", "5", "9"], styletts2_symbol_distance
        )
        assert dict(mapping) == {"1": "0", "8": "9"}

    def test_find_optimal_mapping_leaves_excess_of_a_unpaired(self):
        mapping = find_optimal_mapping(
            ["1", "5", "9"], ["0"], styletts2_symbol_distance
        )
        assert mapping == [("1", "0")]

    def test_find_optimal_mapping_empty_inputs(self):
        assert find_optimal_mapping([], ["a"], styletts2_symbol_distance) == []
        assert find_optimal_mapping(["a"], [], styletts2_symbol_distance) == []

    def test_styletts2_symbol_distance_identical_is_zero(self):
        assert styletts2_symbol_distance("p", "p") == 0.0

    def test_styletts2_symbol_distance_prefers_phonetically_similar_ipa(self):
        assert styletts2_symbol_distance("p", "b") < styletts2_symbol_distance("p", "a")

    def test_styletts2_symbol_distance_non_ipa_graphemes_not_degenerate(self):
        # panphon's feature vectors are all-zero for non-IPA symbols, which
        # would make every grapheme equidistant from every other; the edit
        # distance/unicode fallbacks should not have that problem
        assert styletts2_symbol_distance("1", "2") != styletts2_symbol_distance(
            "1", "22"
        )
        assert styletts2_symbol_distance("1", "2") < styletts2_symbol_distance("1", "4")

    def test_suggest_symbol_mapping_keeps_exact_matches_untouched(self):
        result = suggest_symbol_mapping(["p", "a"], ["p", "a", "b"])
        assert sorted(result.exact) == ["a", "p"]
        assert result.suggestions == {}
        assert result.unmapped == []

    def test_suggest_symbol_mapping_maps_oov_symbol_to_closest_pretrained(self):
        result = suggest_symbol_mapping(["p", "ʒ"], ["p", "ʃ"])
        assert result.exact == ["p"]
        assert result.suggestions == {"ʒ": "ʃ"}
        assert "ʒ" in result.distances

    def test_suggest_symbol_mapping_is_one_to_one(self):
        # two distinct OOV symbols must never be suggested the same
        # pretrained target, since that would collapse them into the same
        # embedding in the frozen pretrained text encoder
        result = suggest_symbol_mapping(["ʒ", "d͡ʒ"], ["p", "ʃ"])
        targets = list(result.suggestions.values())
        assert len(targets) == len(set(targets))

    def test_suggest_symbol_mapping_reports_unmapped_when_no_free_slots(self):
        result = suggest_symbol_mapping(["ʒ", "d͡ʒ", "t͡ʃ"], ["p", "ʃ"])
        assert len(result.suggestions) + len(result.unmapped) == 3
        assert len(result.unmapped) > 0


if __name__ == "__main__":
    main(sys.argv)
