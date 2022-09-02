from unicodedata import normalize
from unittest import TestCase

from config import BaseConfig
from text import TextProcessor


class TextTest(TestCase):
    """Basic test for text input configuration"""

    def setUp(self) -> None:
        self.base_text_processor = TextProcessor(BaseConfig())

    def test_text_to_sequence(self):
        text = "hello world"
        sequence = self.base_text_processor.text_to_sequence(text)
        self.assertEqual(self.base_text_processor.sequence_to_text(sequence), text)

    def test_sequence_to_text(self):
        sequence = [24, 21, 28, 28, 31, 16, 39, 31, 34, 28, 20]
        self.assertEqual(
            self.base_text_processor.text_to_sequence("hello world"), sequence
        )

    def test_cleaners(self):
        text = "hello world"
        text_upper = "HELLO WORLD"
        sequence = self.base_text_processor.text_to_sequence(text_upper)
        self.assertEqual(self.base_text_processor.sequence_to_text(sequence), text)

    def test_phonological_features(self):
        pass

    def test_duplicate_symbols(self):
        duplicate_symbols_text_processor = TextProcessor(
            BaseConfig({"text": {"symbols": {"duplicate": "e"}}})
        )
        self.assertIn("e", duplicate_symbols_text_processor.duplicate_symbols)

    def test_bad_symbol_configuration(self):
        with self.assertRaises(TypeError):
            TextProcessor(BaseConfig({"text": {"symbols": {"bad": 1}}}))

    def test_dipgrahs(self):
        digraph_text_processor = TextProcessor(
            BaseConfig({"text": {"symbols": {"digraph": ["ee"]}}})
        )
        text = "ee"  # should be treated as "ee" and not two instances of "e"
        sequence = digraph_text_processor.text_to_sequence(text)
        self.assertEqual(len(sequence), 1)

    def test_normalization(self):
        # This test doesn't really test very much, but just here to highlight that base cleaning involves NFC
        accented_text_processor = TextProcessor(
            BaseConfig({"text": {"symbols": {"accented": ["é"]}}})
        )
        text = "he\u0301llo world"
        sequence = accented_text_processor.text_to_sequence(text)
        self.assertNotEqual(accented_text_processor.sequence_to_text(sequence), text)
        self.assertEqual(
            accented_text_processor.sequence_to_text(sequence), normalize("NFC", text)
        )

    def test_missing_symbol(self):
        text = "h3llo world"
        sequence = self.base_text_processor.text_to_sequence(text)
        self.assertNotEqual(self.base_text_processor.sequence_to_text(sequence), text)
        self.assertIn("3", self.base_text_processor.missing_symbols)
        self.assertEqual(self.base_text_processor.missing_symbols["3"], 1)
