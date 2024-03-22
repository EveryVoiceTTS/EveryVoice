#!/usr/bin/env python
import string
from pathlib import Path
from typing import Dict, List
from unicodedata import normalize
from unittest import TestCase

from everyvoice.config.text_config import Symbols, TextConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.tests.basic_test_case import BasicTestCase
from everyvoice.text.lookups import build_lookup, lookuptables_from_data
from everyvoice.text.phonemizer import AVAILABLE_G2P_ENGINES, get_g2p_engine
from everyvoice.text.text_processor import TextProcessor
from everyvoice.utils import generic_dict_loader


class TextTest(BasicTestCase):
    """Basic test for text input configuration"""

    def setUp(self) -> None:
        super().setUp()
        self.base_text_processor = TextProcessor(
            TextConfig(symbols=Symbols(letters=string.ascii_letters)),
        )

    def test_text_to_sequence(self):
        text = "hello world"
        sequence = self.base_text_processor.encode_text(text)
        self.assertEqual(self.base_text_processor.decode_tokens(sequence), text)

    def test_token_sequence_to_text(self):
        sequence = [10, 7, 14, 14, 17, 1, 25, 17, 20, 14, 6]
        self.assertEqual(self.base_text_processor.encode_text("hello world"), sequence)

    def test_hardcoded_symbols(self):
        self.assertEqual(
            self.base_text_processor.encode_text("\x80 "),
            [0, 1],
            "pad should be Unicode PAD symbol and index 0, whitespace should be index 1",
        )

    def test_cleaners(self):
        text = "hello world"
        text_upper = "HELLO WORLD"
        sequence = self.base_text_processor.encode_text(text_upper)
        self.assertEqual(self.base_text_processor.decode_tokens(sequence), text)

    def test_punctuation(self):
        text = "hello! How are you? My name's: foo;."
        tokens = self.base_text_processor.apply_tokenization(
            self.base_text_processor.normalize_text(text)
        )
        self.assertEqual(
            self.base_text_processor.apply_punctuation_rules(tokens),
            [
                "h",
                "e",
                "l",
                "l",
                "o",
                "<EXCL>",
                " ",
                "h",
                "o",
                "w",
                " ",
                "a",
                "r",
                "e",
                " ",
                "y",
                "o",
                "u",
                "<QINT>",
                " ",
                "m",
                "y",
                " ",
                "n",
                "a",
                "m",
                "e",
                "<QUOTE>",
                "s",
                "<BB>",
                " ",
                "f",
                "o",
                "o",
                "<BB>",
                "<BB>",
            ],
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
                        "éː",
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
                        "ũ",
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
        moh_text_processor = TextProcessor(moh_config.text)
        normalized_text = moh_text_processor.normalize_text("shéːkon")
        one_hot_tokens = moh_text_processor.encode_text(
            normalized_text, quiet=True
        )  # this finds ː as OOV
        g2p_tokens = moh_text_processor.encode_text(
            normalized_text, lang_id="moh", apply_g2p=True
        )
        feats = moh_text_processor.encode_text(
            normalized_text,
            lang_id="moh",
            apply_g2p=True,
            encode_as_phonological_features=True,
        )
        self.assertEqual(moh_text_processor.decode_tokens(g2p_tokens), "séːɡũ")
        self.assertEqual(len(g2p_tokens), len(feats))
        self.assertNotEqual(len(g2p_tokens), len(one_hot_tokens))
        self.assertEqual(len(feats[0]), moh_config.model.phonological_feats_size)

    def test_duplicate_symbols(self):
        duplicate_symbols_text_processor = TextProcessor(
            TextConfig(symbols=Symbols(letters=string.ascii_letters, duplicate=["e"]))
        )
        self.assertIn("e", duplicate_symbols_text_processor.duplicate_symbols)

    def test_bad_symbol_configuration(self):
        with self.assertRaises(TypeError):
            TextProcessor(
                TextConfig(symbols=Symbols(letters=string.ascii_letters, bad=[1]))
            )

    def test_dipgrahs(self):
        digraph_text_processor = TextProcessor(
            TextConfig(symbols=Symbols(letters=string.ascii_letters, digraph=["ee"]))
        )
        text = "ee"  # should be treated as "ee" and not two instances of "e"
        sequence = digraph_text_processor.encode_text(text)
        self.assertEqual(len(sequence), 1)

    def test_normalization(self):
        # This test doesn't really test very much, but just here to highlight that base cleaning involves NFC
        accented_text_processor = TextProcessor(
            TextConfig(symbols=Symbols(letters=string.ascii_letters, accented=["é"])),
        )
        text = "he\u0301llo world"
        sequence = accented_text_processor.encode_text(text)
        self.assertNotEqual(accented_text_processor.decode_tokens(sequence), text)
        self.assertEqual(
            accented_text_processor.decode_tokens(sequence),
            normalize("NFC", text),
        )

    def test_missing_symbol(self):
        text = "h3llo world"
        sequence = self.base_text_processor.encode_text(text)
        self.assertNotEqual(self.base_text_processor.decode_tokens(sequence), text)
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


class TestG2p(BasicTestCase):
    """Test G2P"""

    def test_many_available_langs(self):
        self.assertGreaterEqual(len(AVAILABLE_G2P_ENGINES), 20)

    def test_basic_g2p(self):
        eng_g2p = get_g2p_engine("eng")
        self.assertEqual(
            eng_g2p("hello world"), ["h", "ʌ", "l", "o", "ʊ", " ", "w", "ɜ˞", "l", "d"]
        )
        # keep's punctuation
        self.assertEqual(
            eng_g2p('hello "world"!!?.'),
            [
                "h",
                "ʌ",
                "l",
                "o",
                "ʊ",
                " ",
                '"',
                "w",
                "ɜ˞",
                "l",
                "d",
                '"',
                "!",
                "!",
                "?",
                ".",
            ],
        )
        # another language
        str_g2p = get_g2p_engine("str")
        self.assertEqual(
            str_g2p("SENĆOŦEN"), ["s", "ʌ", "n", "t͡ʃ", "ɑ", "θ", "ʌ", "n"]
        )
        # test lang_id missing
        with self.assertRaises(NotImplementedError):
            get_g2p_engine("boop")
