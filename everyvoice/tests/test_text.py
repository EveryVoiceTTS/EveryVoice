from unicodedata import normalize
from unittest import TestCase, main

from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.text import TextProcessor


class TextTest(TestCase):
    """Basic test for text input configuration"""

    def setUp(self) -> None:
        self.base_text_processor = TextProcessor(
            EveryVoiceConfig.load_config_from_path().aligner
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
        moh_config = (
            EveryVoiceConfig.load_config_from_path().feature_prediction.update_config(
                {
                    "text": {
                        "symbols": {
                            "letters": [
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
                        }
                    }
                }
            )
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
            EveryVoiceConfig.load_config_from_path().feature_prediction.update_config(
                ({"text": {"symbols": {"duplicate": "e"}}})
            )
        )
        self.assertIn("e", duplicate_symbols_text_processor.duplicate_symbols)

    def test_bad_symbol_configuration(self):
        with self.assertRaises(TypeError):
            TextProcessor(
                EveryVoiceConfig.load_config_from_path().feature_prediction.update_config(
                    ({"text": {"symbols": {"bad": 1}}})
                )
            )

    def test_dipgrahs(self):
        digraph_text_processor = TextProcessor(
            EveryVoiceConfig.load_config_from_path().feature_prediction.update_config(
                ({"text": {"symbols": {"digraph": ["ee"]}}})
            )
        )
        text = "ee"  # should be treated as "ee" and not two instances of "e"
        sequence = digraph_text_processor.text_to_sequence(text)
        self.assertEqual(len(sequence), 1)

    def test_normalization(self):
        # This test doesn't really test very much, but just here to highlight that base cleaning involves NFC
        accented_text_processor = TextProcessor(
            EveryVoiceConfig.load_config_from_path().feature_prediction.update_config(
                ({"text": {"symbols": {"accented": ["é"]}}})
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


if __name__ == "__main__":
    main()
