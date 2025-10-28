#!/usr/bin/env python

import string
from pathlib import Path
from unicodedata import normalize
from unittest import TestCase, main

from pydantic import ValidationError

from everyvoice import exceptions
from everyvoice.config.text_config import Punctuation, Symbols, TextConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.tests.stubs import TEST_CONTACT, silence_c_stderr
from everyvoice.text.features import N_PHONOLOGICAL_FEATURES
from everyvoice.text.lookups import build_lookup, lookuptables_from_data
from everyvoice.text.text_processor import JOINER_SUBSTITUTION, TextProcessor
from everyvoice.text.textsplit import chunk_text
from everyvoice.text.utils import is_sentence_final
from everyvoice.utils import (
    collapse_whitespace,
    generic_psv_filelist_reader,
    lower,
    nfc_normalize,
)


class TextTest(TestCase):
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
            contact=TEST_CONTACT,
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

    def test_is_sentence_final(self):
        self.assertTrue(is_sentence_final("!"))
        self.assertTrue(is_sentence_final("?"))
        self.assertTrue(is_sentence_final("."))
        self.assertTrue(is_sentence_final("᙮"))
        self.assertFalse(is_sentence_final("¡"))
        self.assertFalse(is_sentence_final("¿"))


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

        def remove_language(data: list[dict[str, str]]) -> list[dict[str, str]]:
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

        def remove_speaker(data: list[dict[str, str]]) -> list[dict[str, str]]:
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


class TestTextSplit(TestCase):
    def test_strong_boundary(self):
        a = "There are approximately 70 Indigenous languages spoken in Canada from 10 distinct language families."
        b = "As a consequence of the residential school system and other policies of cultural suppression, the majority of these languages now have fewer than 500 fluent speakers remaining, most of them elderly."
        text = a + " " + b
        self.assertEqual([a, b], chunk_text(text))

    def test_weak_boundary(self):
        a = "There are approximately 70 Indigenous languages spoken in Canada from 10 distinct language families; as a consequence of the residential school system and other policies of cultural suppression,"
        b = "the majority of these languages now have fewer than 500 fluent speakers remaining, most of them elderly."
        text = a + " " + b
        self.assertEqual([a, b], chunk_text(text))

    def test_custom_desired_length(self):
        a = "There are approximately 70 Indigenous languages spoken in Canada!"
        b = "Among these, there are 10 distinct language families."
        c = "As a consequence of the residential school system and other policies of cultural suppression,"
        d = "the majority of these languages now have fewer than 500 fluent speakers remaining."
        e = "Most fluent speakers are elderly."
        text = a + " " + b + " " + c + " " + d + " " + e
        self.assertEqual([a + " " + b, c + " " + d, e], chunk_text(text, 75, 1000))

    def test_normalization(self):
        a = "Welcome to the EveryVoice Documentation! Please read the background section below."
        text = "       Welcome to     the EveryVoice       Documentation!\n\n\n\nPlease read the background section below.                        "
        self.assertEqual([a], chunk_text(text))

    def test_quote_toggling(self):
        text = 'There are approximately "70 Indigenous languages spoken in Canada. The majority of these languages" now have fewer than 500 fluent speakers remaining.'
        self.assertEqual([text], chunk_text(text, 75, 1000))

    def test_invalid_lengths(self):
        text = "Hello, world!"
        with self.assertRaises(AssertionError):
            chunk_text(text, 200, 100)

    def test_no_boundaries(self):
        """
        When there are no boundaries, chunk_text should split at the max length (possibly in the middle of a word).
        """
        a = "There are approximately 70 Indigenous languages spoken in Canada from 10 distinct language families. As a consequence of the residential school system and other policies of cultural suppression, the m"
        b = "ajority of these languages now have fewer than 500 fluent speakers remaining, most of them elderly."
        text = a + b
        self.assertEqual(
            [a, b], chunk_text(text, weak_boundaries="", strong_boundaries="")
        )

    def test_custom_weak_boundaries(self):
        """
        Test that a split DOES NOT occur on the SENĆOŦEN , character.
        """
        # This text, in SENĆOŦEN, is the W̱SÁNEĆ Mission Statement (https://wsanecschoolboard.ca/sencoten-language/)
        a = "W̱UĆIST TŦE SKÁLs I,"  # This sentence is intentionally broken up mid-word
        b = "TŦE Ś,X̱ENAṈs ĆSE LÁ,E TŦE ÁLEṈENEȻ TŦE W̱SÁNEĆ."
        text = a + " " + b
        # With custom weak boundaries
        self.assertNotIn(a, chunk_text(text, 15, 30, weak_boundaries=":;"))
        # Without custom weak boundaries
        self.assertIn(a, chunk_text(text, 15, 30))

    def test_custom_strong_boundaries(self):
        """
        Test that the a split occurs on ᙮, the Cree full stop.
        """
        # This text, in East Cree, is from the 'Marriage and Inuits in the old days' story (https://www.eastcree.org/cree/en/stories/)
        a = "ᐧᐋᔥᑭᒡ ᐃᓐᑖᐦ ᑖᐹ ᐧᐃᒡ ᓃᔓᑳᐳᐧᐃᒡ ᐊᐧᐋᓂᒌ ᒥᒄ ᒌᐦ ᐧᐄᒋᒥᑑᒡ ᐋᑳ ᑭᐧᐹ ᐧᐃᒡ ᑖᑦ ᐋᔨᒻᐦᐋᐅᒋᒫᐤ᙮"
        b = "ᐄᔥᒋᒫᐅᒡ ᒌᐦ ᓂᐱᐦᐋᐅᒡ ᐄᔨᔨᐤᐦ ᒥᒄ ᒌᐦ ᑖᐤ ᐹᔨᒄ ᐄᔥᒌᒫᐤ ᐋᑳ ᐧᐃᒡ ᒦᐧᔮᔨᑎ ᐋᐦ ᓂᐱᐦᐄᐧᐋᑦ ᑳᐦ ᐧᐄᑎᒥᐧᐋᑦ ᐋᓂᔮᐦ ᐄᔨᔨᐤᐦ ᐋᐃᑖᔨᑎᒦᒡ ᐄᔥᒋᒫᐤ ᒑᓂᐱᐦᐋᔨᒡ ᑳᓂᑎᐧᐋᔨᑎᒧᐧᐋᑦ ᒋᔥᑖᒫᐤ᙮"
        text = a + " " + b
        # Behaviour with custom strong boundary
        self.assertEqual([a, b], chunk_text(text, 50, 200, strong_boundaries="᙮"))
        # Without custom strong boundary
        self.assertEqual([text], chunk_text(text, 50, 200))


if __name__ == "__main__":
    main()
