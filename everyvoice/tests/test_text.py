import string
from pathlib import Path
from typing import Dict, List
from unicodedata import normalize
from unittest import TestCase

from pydantic import ValidationError

import everyvoice
from everyvoice import exceptions
from everyvoice.config.text_config import Punctuation, Symbols, TextConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.tests.basic_test_case import BasicTestCase
from everyvoice.tests.stubs import silence_c_stderr
from everyvoice.text.features import N_PHONOLOGICAL_FEATURES
from everyvoice.text.lookups import build_lookup, lookuptables_from_data
from everyvoice.text.phonemizer import (
    AVAILABLE_G2P_ENGINES,
    DEFAULT_G2P,
    CachingG2PEngine,
    get_g2p_engine,
    make_default_g2p_engines,
)
from everyvoice.text.text_processor import JOINER_SUBSTITUTION, TextProcessor
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

    def test_text_to_sequence(self):
        text = "hello world"
        sequence = self.base_text_processor.encode_text(text)
        self.assertEqual(self.base_text_processor.decode_tokens(sequence, "", ""), text)

    def test_token_sequence_to_text(self):
        sequence = [60, 57, 64, 64, 67, 1, 75, 67, 70, 64, 56]
        self.assertEqual(self.base_text_processor.encode_text("hello world"), sequence)

    def test_hardcoded_symbols(self):
        self.assertEqual(
            self.base_text_processor.encode_text("\x80 \x80"),
            [0, 1, 0],
            "pad should be Unicode PAD symbol and index 0, whitespace should be index 1",
        )

    def test_cleaners_with_upper(self):
        text = "hello world"
        text_upper = "HELLO WORLD"
        with silence_c_stderr():
            upper_text_processor = TextProcessor(
                TextConfig(
                    cleaners=[collapse_whitespace, lower],
                    symbols=Symbols(letters=list(string.ascii_letters)),
                ),
            )
        sequence = upper_text_processor.encode_text(text_upper)
        self.assertEqual(upper_text_processor.decode_tokens(sequence, "", ""), text)

    def test_no_duplicate_punctuation(self):
        with self.assertRaises(ValidationError):
            TextConfig(symbols=Symbols(letters=[":"] + list(string.ascii_letters)))

    def test_punctuation(self):
        text = "hello! How are you? My name's: foo;."
        with silence_c_stderr():
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
                "<COLON>",
                " ",
                "f",
                "o",
                "o",
                "<SEMICOL>",
                "<PERIOD>",
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
        self.assertEqual(moh_text_processor.decode_tokens(g2p_tokens, "", ""), "séːɡũ")
        self.assertEqual(len(g2p_tokens), len(feats))
        self.assertNotEqual(len(g2p_tokens), len(one_hot_tokens))
        self.assertEqual(len(feats[0]), N_PHONOLOGICAL_FEATURES)

    def test_duplicates_removed(self):
        duplicate_symbols_text_processor = TextProcessor(
            TextConfig(
                symbols=Symbols(letters=list(string.ascii_letters), duplicate=["e"])
            )
        )
        self.assertEqual(
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
        self.assertNotEqual(
            accented_text_processor.decode_tokens(sequence, "", ""), text
        )
        self.assertEqual(
            accented_text_processor.decode_tokens(sequence, "", ""),
            normalize("NFC", text),
        )
        self.assertNotEqual(
            self.base_text_processor.apply_cleaners(text), normalize("NFC", text)
        )

    def test_missing_symbol(self):
        text = "h3llo world"
        with silence_c_stderr():
            sequence = self.base_text_processor.encode_text(text)
        self.assertNotEqual(self.base_text_processor.decode_tokens(sequence), text)
        self.assertIn("3", self.base_text_processor.missing_symbols)
        self.assertEqual(self.base_text_processor.missing_symbols["3"], 1)

    def test_use_slash(self):
        text = "word/token"
        text_processor = TextProcessor(
            TextConfig(symbols=Symbols(letters=list(string.ascii_letters) + ["/"])),
        )
        sequence = text_processor.encode_text(text)
        decoded = text_processor.decode_tokens(sequence)
        self.assertEqual(decoded, "w/o/r/d/" + JOINER_SUBSTITUTION + "/t/o/k/e/n")
        encoded = text_processor.encode_escaped_string_sequence(decoded)
        self.assertEqual(encoded, sequence)

        with self.assertRaises(exceptions.OutOfVocabularySymbolError):
            # / is OOV, so JOINER_SUBSTITUTION will also be OOV
            self.base_text_processor.encode_escaped_string_sequence(decoded)

    def test_encode_string_tokens(self):
        self.assertEqual(
            self.base_text_processor.encode_string_tokens(["a", "b", ",", " ", "c"]),
            self.base_text_processor.encode_escaped_string_sequence("a/b/,/ /c"),
        )
        with self.assertRaises(exceptions.OutOfVocabularySymbolError):
            self.base_text_processor.encode_string_tokens(["oov"])
        with self.assertRaises(exceptions.OutOfVocabularySymbolError):
            self.base_text_processor.encode_string_tokens([JOINER_SUBSTITUTION])


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

    def setUp(self) -> None:
        super().setUp()
        # Each test in this suite needs to start with a fresh, empty
        # AVAILABLE_G2P_ENGINES cache, otherwise caching due to previous calls
        # get get_g2p_engine() in other suites could invalidate some tests here.
        self.SAVED_AVAILABLE_G2P_ENGINES = dict(AVAILABLE_G2P_ENGINES)
        AVAILABLE_G2P_ENGINES.clear()
        AVAILABLE_G2P_ENGINES.update(make_default_g2p_engines())

    def tearDown(self) -> None:
        super().setUp()
        AVAILABLE_G2P_ENGINES.clear()
        AVAILABLE_G2P_ENGINES.update(self.SAVED_AVAILABLE_G2P_ENGINES)

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
        self.assertEqual(moh_g2p("\u00e9"), ["\u00e9"])

    def test_invalid_lang_id(self):
        """
        User asked for a language that is not supported by AVAILABLE_G2P_ENGINES.
        """
        lang_id = "unittest"
        self.assertNotIn(lang_id, AVAILABLE_G2P_ENGINES)
        with self.assertRaisesRegex(
            NotImplementedError,
            rf"Sorry, we don't have a grapheme-to-phoneme engine available for {lang_id}.*",
            msg="The user provided G2P engine shouldn't be available before loading a TextConfig.",
        ):
            get_g2p_engine(lang_id)

    def test_custom_g2p_engine(self):
        """
        Use a user provided G2P engine.
        """
        lang_id = "unittest"
        with self.assertRaisesRegex(
            NotImplementedError,
            rf"Sorry, we don't have a grapheme-to-phoneme engine available for {lang_id}.*",
            msg="The user provided G2P engine shouldn't be available before loading a TextConfig.",
        ):
            get_g2p_engine(lang_id)
        TextConfig(g2p_engines={lang_id: "everyvoice.tests.g2p_engines.valid"})
        self.assertIn(lang_id, AVAILABLE_G2P_ENGINES)
        self.assertIs(
            AVAILABLE_G2P_ENGINES[lang_id],
            everyvoice.tests.g2p_engines.valid,
        )

    def test_invalid_g2p_engine(self):
        """
        The only string value allowed in AVAILABLE_G2P_ENGINES is 'DEFAULT_G2P'.
        """

        lang_id = "unittest"
        AVAILABLE_G2P_ENGINES[lang_id] = "WRONG"
        with self.assertRaisesRegex(
            AssertionError,
            f"Internal error: the only str value allowed in AVAILABLE_G2P_ENGINES is '{DEFAULT_G2P}'.",
        ):
            get_g2p_engine(lang_id)

    def test_autoload(self):
        """
        Default G2PEngine should autoload a CachingG2PEngine(lang_id).
        """
        lang_id = "eng"
        self.assertIn(lang_id, AVAILABLE_G2P_ENGINES)
        self.assertEqual(AVAILABLE_G2P_ENGINES[lang_id], DEFAULT_G2P)

        g2p_engine = get_g2p_engine(lang_id)
        self.assertFalse(isinstance(g2p_engine, str))
        self.assertTrue(isinstance(g2p_engine, CachingG2PEngine))


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
                "]",
                "}",
                "[",
                ")",
                "*",
                "{",
                "(",
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
