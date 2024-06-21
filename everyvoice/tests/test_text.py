import doctest
import string
from pathlib import Path
from typing import Dict, List
from unicodedata import normalize
from unittest import TestCase

from pydantic import ValidationError

import everyvoice.text.utils
from everyvoice.config.text_config import Punctuation, Symbols, TextConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.tests.basic_test_case import BasicTestCase
from everyvoice.text.features import N_PHONOLOGICAL_FEATURES
from everyvoice.text.lookups import build_lookup, lookuptables_from_data
from everyvoice.text.phonemizer import AVAILABLE_G2P_ENGINES, get_g2p_engine
from everyvoice.text.text_processor import TextProcessor
from everyvoice.utils import (
    collapse_whitespace,
    generic_psv_filelist_reader,
    lower,
    nfc_normalize,
)


class TextTest(BasicTestCase):
    """Basic test for text input configuration"""

    def setUp(self) -> None:
        super().setUp()
        self.base_text_processor = TextProcessor(
            TextConfig(symbols=Symbols(letters=list(string.ascii_letters))),
        )

    def test_run_doctest(self):
        """Run doctests in everyvoice.utils"""
        results = doctest.testmod(everyvoice.text)
        self.assertFalse(results.failed, results)

    def test_text_to_sequence(self):
        text = "hello world"
        sequence = self.base_text_processor.encode_text(text)
        self.assertEqual(self.base_text_processor.decode_tokens(sequence, ""), text)

    def test_token_sequence_to_text(self):
        sequence = [51, 48, 55, 55, 58, 1, 66, 58, 61, 55, 47]
        self.assertEqual(self.base_text_processor.encode_text("hello world"), sequence)

    def test_hardcoded_symbols(self):
        self.assertEqual(
            self.base_text_processor.encode_text("\x80 "),
            [0, 1],
            "pad should be Unicode PAD symbol and index 0, whitespace should be index 1",
        )

    def test_cleaners_with_upper(self):
        text = "hello world"
        text_upper = "HELLO WORLD"
        upper_text_processor = TextProcessor(
            TextConfig(
                cleaners=[collapse_whitespace, lower],
                symbols=Symbols(letters=list(string.ascii_letters)),
            ),
        )
        sequence = upper_text_processor.encode_text(text_upper)
        self.assertEqual(upper_text_processor.decode_tokens(sequence, ""), text)

    def test_punctuation(self):
        text = "hello! How are you? My name's: foo;."
        upper_text_processor = TextProcessor(
            TextConfig(
                cleaners=[collapse_whitespace, lower],
                symbols=Symbols(letters=list(string.ascii_letters)),
            ),
        )
        tokens = upper_text_processor.apply_tokenization(
            upper_text_processor.normalize_text(text)
        )
        self.assertEqual(
            upper_text_processor.apply_punctuation_rules(tokens),
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
                cleaners=[collapse_whitespace, lower, nfc_normalize],
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
                ),
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
        self.assertEqual(moh_text_processor.decode_tokens(g2p_tokens, ""), "séːɡũ")
        self.assertEqual(len(g2p_tokens), len(feats))
        self.assertNotEqual(len(g2p_tokens), len(one_hot_tokens))
        self.assertEqual(len(feats[0]), N_PHONOLOGICAL_FEATURES)

    def test_duplicates_removed(self):
        duplicate_symbols_text_processor = TextProcessor(
            TextConfig(
                symbols=Symbols(letters=list(string.ascii_letters), duplicate=["e"])
            )
        )
        self.assertEquals(
            len([x for x in duplicate_symbols_text_processor.symbols if x == "e"]), 1
        )

    def test_bad_symbol_configuration(self):
        with self.assertRaises(ValidationError):
            TextProcessor(
                TextConfig(symbols=Symbols(letters=list(string.ascii_letters), bad=[1]))
            )

    def test_dipgrahs(self):
        digraph_text_processor = TextProcessor(
            TextConfig(
                symbols=Symbols(letters=list(string.ascii_letters), digraph=["ee"])
            )
        )
        text = "ee"  # should be treated as "ee" and not two instances of "e"
        sequence = digraph_text_processor.encode_text(text)
        self.assertEqual(len(sequence), 1)

    def test_normalization(self):
        # This test doesn't really test very much, but just here to highlight that base cleaning doesn't involve NFC
        accented_text_processor = TextProcessor(
            TextConfig(
                cleaners=[nfc_normalize],
                symbols=Symbols(letters=list(string.ascii_letters), accented=["é"]),
            ),
        )
        text = "he\u0301llo world"
        sequence = accented_text_processor.encode_text(text)
        self.assertNotEqual(accented_text_processor.decode_tokens(sequence, ""), text)
        self.assertEqual(
            accented_text_processor.decode_tokens(sequence, ""),
            normalize("NFC", text),
        )
        self.assertNotEqual(
            self.base_text_processor.apply_cleaners(text), normalize("NFC", text)
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
                generic_psv_filelist_reader(base_path / "training_filelist.psv"),
                generic_psv_filelist_reader(base_path / "validation_filelist.psv"),
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
                    generic_psv_filelist_reader(base_path / "training_filelist.psv")
                ),
                remove_language(
                    generic_psv_filelist_reader(base_path / "validation_filelist.psv")
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
                    generic_psv_filelist_reader(base_path / "training_filelist.psv")
                ),
                remove_speaker(
                    generic_psv_filelist_reader(base_path / "validation_filelist.psv")
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


class TestG2p(TestCase):
    """Test G2P"""

    def test_many_available_langs(self):
        self.assertGreaterEqual(len(AVAILABLE_G2P_ENGINES), 20)

    def test_pua_chars(self):
        eng_g2p = get_g2p_engine("eng")
        und_g2p = get_g2p_engine("und")
        tokens = eng_g2p("h_e_l_l_o")
        self.assertEqual(
            tokens,
            ["e", "ɪ", "t", "ʃ", "_", "i", "_", "ɛ", "l", "_", "ɛ", "l", "_", "o", "ʊ"],
        )
        tokens = und_g2p("___")
        self.assertEqual(tokens, ["_", "_", "_"])

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
        self.assertEqual(str_g2p("SENĆOŦEN"), ["s", "ʌ", "n", "t͡ʃ", "ɑ", "θ", "ʌ", "n"])
        # test lang_id missing
        with self.assertRaises(NotImplementedError):
            get_g2p_engine("boop")

    def test_phonemizer_normalization(self):
        moh_g2p = get_g2p_engine("moh")
        self.assertEqual(moh_g2p("\u00E9"), ["\u00E9"])


class PunctuationTest(TestCase):
    def test_all(self):
        """Make sure we get the union of all punctuation characters when calling `all`."""
        punctuation = Punctuation()
        self.assertSetEqual(
            punctuation.all,
            {
                "?",
                "¿",
                "!",
                "¡",
                ",",
                ";",
                '"',
                "'",
                "«",
                "”",
                ":",
                "»",
                "-",
                "—",
                ".",
                "“",
                "…",
            },
        )


class SymbolsTest(TestCase):
    def test_all_except_punctuation(self):
        """Not withstanding the random new member variables defined by the
        user, we should get the union of them excluding what is in
        `punctuation`.
        """
        symbols = Symbols(
            dataset1=["a", "b"],
            dataset2=["X", "Y", "Z"],
        )
        self.assertSetEqual(
            symbols.all_except_punctuation, {"a", "b", "X", "Y", "Z", "<SIL>"}
        )
