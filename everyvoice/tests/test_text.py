#!/usr/bin/env python
import string
from pathlib import Path
from typing import Dict, List
from unicodedata import normalize
from unittest import TestCase

from everyvoice.config.text_config import Symbols, TextConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.tests.basic_test_case import BasicTestCase
from everyvoice.text import TextProcessor
from everyvoice.text.lookups import build_lookup, lookuptables_from_data
from everyvoice.utils import generic_dict_loader


class TextTest(BasicTestCase):
    """Basic test for text input configuration"""

    def setUp(self) -> None:
        super().setUp()
        self.base_text_processor = TextProcessor(
            FeaturePredictionConfig(
                contact=self.contact,
                text=TextConfig(symbols=Symbols(letters=string.ascii_letters)),
            )
        )

    def test_text_to_sequence(self):
        text = "hello world"
        sequence = self.base_text_processor.text_to_sequence(text)
        self.assertEqual(
            self.base_text_processor.token_sequence_to_text(sequence), text
        )

    def test_token_sequence_to_text(self):
        sequence = [27, 24, 31, 31, 34, 19, 42, 34, 37, 31, 23]
        self.assertEqual(
            self.base_text_processor.text_to_sequence("hello world"), sequence
        )

    def test_cleaners(self):
        text = "hello world"
        text_upper = "HELLO WORLD"
        sequence = self.base_text_processor.text_to_sequence(text_upper)
        self.assertEqual(
            self.base_text_processor.token_sequence_to_text(sequence), text
        )

    def test_phonological_features(self):
        moh_config = FeaturePredictionConfig(
            contact=self.contact,
            text=TextConfig(
                symbols=Symbols(
                    letters=[
                        "ʌ̃̀ː",
                        "ʌ̃́ː",
                        "t͡ʃ",
                        "d͡ʒ",
                        "ʌ̃́",
                        "ʌ̃ː",
                        "kʰʷ",
                        "ũ̀ː",
                        "ɡʷ",
                        "áː",
                        "àː",
                        "aː",
                        "ʌ̃",
                        "èː",
                        "éː",
                        "iː",
                        "íː",
                        "ìː",
                        "kʷ",
                        "ṹː",
                        "óː",
                        "òː",
                        "ʃ",
                        "d",
                        "ɡ",
                        "á",
                        "a",
                        "é",
                        "e",
                        "í",
                        "i",
                        "k",
                        "n",
                        "ṹ",
                        "ũ",
                        "ó",
                        "o",
                        "r",
                        "h",
                        "t",
                        "s",
                        "w",
                        "f",
                        "j",
                        "ʔ",
                    ]
                )
            ),
        )
        moh_text_processor = TextProcessor(moh_config)
        tokens = moh_text_processor.text_to_tokens("shéːkon")
        feats = moh_text_processor.text_to_phonological_features("shéːkon")
        self.assertEqual(len(tokens), len(feats))
        self.assertEqual(len(feats[0]), moh_config.model.phonological_feats_size)
        extra_tokens = moh_text_processor.text_to_tokens("shéːkon7")
        extra_feats = moh_text_processor.text_to_phonological_features("shéːkon7")
        self.assertEqual(len(feats), len(extra_feats))
        self.assertEqual(len(extra_feats), len(extra_tokens))

    def test_duplicate_symbols(self):
        duplicate_symbols_text_processor = TextProcessor(
            FeaturePredictionConfig(
                contact=self.contact,
                text=TextConfig(
                    symbols=Symbols(letters=string.ascii_letters, duplicate=["e"])
                ),
            )
        )
        self.assertIn("e", duplicate_symbols_text_processor.duplicate_symbols)

    def test_bad_symbol_configuration(self):
        with self.assertRaises(TypeError):
            TextProcessor(
                FeaturePredictionConfig(
                    contact=self.contact,
                    text=TextConfig(
                        symbols=Symbols(letters=string.ascii_letters, bad=[1])
                    ),
                )
            )

    def test_dipgrahs(self):
        digraph_text_processor = TextProcessor(
            FeaturePredictionConfig(
                contact=self.contact,
                text=TextConfig(
                    symbols=Symbols(letters=string.ascii_letters, digraph=["ee"])
                ),
            )
        )
        text = "ee"  # should be treated as "ee" and not two instances of "e"
        sequence = digraph_text_processor.text_to_sequence(text)
        self.assertEqual(len(sequence), 1)

    def test_normalization(self):
        # This test doesn't really test very much, but just here to highlight that base cleaning involves NFC
        accented_text_processor = TextProcessor(
            FeaturePredictionConfig(
                contact=self.contact,
                text=TextConfig(
                    symbols=Symbols(letters=string.ascii_letters, accented=["é"])
                ),
            )
        )
        text = "he\u0301llo world"
        sequence = accented_text_processor.text_to_sequence(text)
        self.assertNotEqual(
            accented_text_processor.token_sequence_to_text(sequence), text
        )
        self.assertEqual(
            accented_text_processor.token_sequence_to_text(sequence),
            normalize("NFC", text),
        )

    def test_missing_symbol(self):
        text = "h3llo world"
        sequence = self.base_text_processor.text_to_sequence(text)
        self.assertNotEqual(
            self.base_text_processor.token_sequence_to_text(sequence), text
        )
        self.assertIn("3", self.base_text_processor.missing_symbols)
        self.assertEqual(self.base_text_processor.missing_symbols["3"], 1)


class LookupTableTest(TestCase):
    def test_build_lookup(self):
        """Make sure the original order of the keys is preserved"""
        key = "speaker"
        data = [
            {key: "Samuel"},
            {key: "Eric"},
            {key: "Eric"},
            {key: "Marc"},
            {key: "Aidan"},
            {key: "Marc"},
            {key: "Samuel"},
        ]
        speaker2id = build_lookup(data, key)
        self.assertDictEqual(
            speaker2id,
            {
                "Samuel": 0,
                "Eric": 1,
                "Marc": 2,
                "Aidan": 3,
            },
        )


class LookupTablesTest(TestCase):
    def test_lookuptables_from_data(self):
        """
        Text looluptables for a multilangual and multispeaker.
        """
        base_path = Path(__file__).parent / "data/lookuptable/"
        lang2id, speaker2id = lookuptables_from_data(
            (
                generic_dict_loader(base_path / "training_filelist.psv"),
                generic_dict_loader(base_path / "validation_filelist.psv"),
            )
        )
        self.assertDictEqual(
            lang2id, {"crk": 0, "git": 1, "str": 2}, "Language lookup tables differ"
        )
        self.assertDictEqual(
            speaker2id,
            {"0": 0, "1": 1, "2": 2, "3": 3},
            "Speaker lookup tables differ.",
        )

    def test_no_language(self):
        """
        Test a datasest that has no language.
        """

        def remove_language(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
            for d in data:
                del d["language"]
            return data

        base_path = Path(__file__).parent / "data/lookuptable/"
        lang2id, speaker2id = lookuptables_from_data(
            (
                remove_language(
                    generic_dict_loader(base_path / "training_filelist.psv")
                ),
                remove_language(
                    generic_dict_loader(base_path / "validation_filelist.psv")
                ),
            )
        )
        self.assertDictEqual(lang2id, {}, "Language lookup tables differ")
        self.assertDictEqual(
            speaker2id,
            {"0": 0, "1": 1, "2": 2, "3": 3},
            "Speaker lookup tables differ.",
        )

    def test_no_speaker(self):
        """
        Test a datasest that has no speaker.
        """

        def remove_speaker(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
            for d in data:
                del d["speaker"]
            return data

        base_path = Path(__file__).parent / "data/lookuptable/"
        lang2id, speaker2id = lookuptables_from_data(
            (
                remove_speaker(
                    generic_dict_loader(base_path / "training_filelist.psv")
                ),
                remove_speaker(
                    generic_dict_loader(base_path / "validation_filelist.psv")
                ),
            )
        )
        self.assertDictEqual(
            lang2id, {"crk": 0, "git": 1, "str": 2}, "Language lookup tables differ"
        )
        self.assertDictEqual(
            speaker2id,
            {},
            "Speaker lookup tables differ.",
        )
