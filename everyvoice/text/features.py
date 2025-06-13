import numpy as np
import numpy.typing as npt
from panphon import FeatureTable

from everyvoice.config.text_config import TextConfig

N_PHONOLOGICAL_FEATURES = 43
DEFAULT_PUNCTUATION_HASH = {
    "exclamations": "<EXCL>",
    "ellipses": "<EPS>",
    "question_symbols": "<QINT>",
    "quotemarks": "<QUOTE>",
    "periods": "<PERIOD>",
    "commas": "<COMMA>",
    "colons": "<COLON>",
    "semi_colons": "<SEMICOL>",
    "hyphens": "<HYPHEN>",
    "parentheses": "<PAREN>",
}


class PhonologicalFeatureCalculator:
    def __init__(
        self, text_config: TextConfig, punctuation_hash: dict = DEFAULT_PUNCTUATION_HASH
    ):
        self.config = text_config
        self.punctuation_hash = punctuation_hash
        self.feature_table = FeatureTable()

    def mask_token(self):
        return self.get_features(["[MASK]"])[0]

    def pad_token(self):
        return self.get_features(["[PAD]"])[0]

    def cls_token(self):
        return self.get_features(["[CLS]"])[0]

    def sep_token(self):
        return self.get_features(["[SEP]"])[0]

    def unk_token(self):
        return self.get_features(["[UNK]"])[0]

    def get_punctuation_features(self, tokens: list[str]) -> npt.NDArray[np.float32]:
        """Get Punctuation features.
           One-hot encodes the allowable types of punctuation and returns zeros elsewhere

        Args:
            tokens (list[str]): a list of IPA and normalized punctuation tokens

        Returns:
            npt.NDArray[np.float32]: a seven-dimensional one-hot encoding of punctuation, white space and silence

        >>> pf = PhonologicalFeatureCalculator(TextConfig())
        >>> pf.get_punctuation_features(['h', 'ʌ', 'l', 'o', 'ʊ', '<EXCL>'])
        array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]], dtype=float32)
        """
        punctuation_features = []
        for char in tokens:
            if char == " ":
                punctuation_features.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif char == self.punctuation_hash["question_symbols"]:
                punctuation_features.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif char == self.punctuation_hash["periods"]:
                punctuation_features.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif char == self.punctuation_hash["colons"]:
                punctuation_features.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            elif char == self.punctuation_hash["semi_colons"]:
                punctuation_features.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
            elif char == self.punctuation_hash["commas"]:
                punctuation_features.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            elif char == self.punctuation_hash["hyphens"]:
                punctuation_features.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
            elif char == self.punctuation_hash["quotemarks"]:
                punctuation_features.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
            elif char == self.punctuation_hash["parentheses"]:
                punctuation_features.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            elif char == self.punctuation_hash["ellipses"]:
                punctuation_features.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            elif char == self.punctuation_hash["exclamations"]:
                punctuation_features.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
            elif char in self.config.symbols.silence:
                punctuation_features.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
            else:
                punctuation_features.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        return np.array(punctuation_features, dtype=np.float32)

    def get_stress_features(self, tokens: list[str]) -> npt.NDArray[np.float32]:
        """Get stress features
           Can be either primary ˈ or secondary stress ˌ

        Args:
            tokens (list[str]): a list of IPA and normalized punctuation tokens

        Returns:
            npt.NDArray[np.float32]: a two-dimensional one-hot encoding of primary and secondary stress

        >>> pf = PhonologicalFeatureCalculator(TextConfig())
        >>> pf.get_stress_features(['ˈ', 'ˌ' ])
        array([[1., 0.],
               [0., 1.]], dtype=float32)
        """
        stress_features = []
        for char in tokens:
            if char == "ˈ":
                stress_features.append([1, 0])
            elif char == "ˌ":
                stress_features.append([0, 1])
            else:
                stress_features.append([0, 0])
        return np.array(stress_features, dtype=np.float32)

    def get_special_token_features(self, tokens: list[str]) -> npt.NDArray[np.float32]:
        """Get special token features
           Can be \x80 (PAD symbol), [UNK], [CLS], [SEP], or [MASK]

        Args:
            tokens (list[str]): a list of IPA and normalized punctuation tokens

        Returns:
            npt.NDArray[np.float32]: a five-dimensional one-hot encoding of special tokens

        >>> pf = PhonologicalFeatureCalculator(TextConfig())
        >>> pf.get_special_token_features(['\x80', '[MASK]', '[CLS]', '[SEP]', '[UNK]' ])
        array([[1., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0.],
               [0., 0., 1., 0., 0.],
               [0., 0., 0., 1., 0.],
               [0., 0., 0., 0., 1.]], dtype=float32)
        """
        special_token_features = []
        for char in tokens:
            if char in ["\x80", "[PAD]"]:
                special_token_features.append([1, 0, 0, 0, 0])
            elif char == "[MASK]":
                special_token_features.append([0, 1, 0, 0, 0])
            elif char == "[CLS]":
                special_token_features.append([0, 0, 1, 0, 0])
            elif char == "[SEP]":
                special_token_features.append([0, 0, 0, 1, 0])
            elif char == "[UNK]":
                special_token_features.append([0, 0, 0, 0, 1])
            else:
                special_token_features.append([0, 0, 0, 0, 0])
        return np.array(special_token_features, dtype=np.float32)

    def token_to_segmental_features(self, token: str) -> npt.NDArray[np.float32]:
        """Turn a token to a feature vector with panphon

        Args:
            token (str): a token to convert to segmental features

        Returns:
            npt.NDArray[np.float32]: a list of place and manner of articulation feature values

        >>> pf = PhonologicalFeatureCalculator(TextConfig())
        >>> pf.token_to_segmental_features('\x80') # pad symbol is all zeros
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0.], dtype=float32)
        >>> pf.token_to_segmental_features('*') # punctuation is all zeros
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0.], dtype=float32)
        >>> pf.token_to_segmental_features('ʌ')
        array([ 1.,  1., -1.,  1., -1., -1., -1.,  0.,  1., -1., -1.,  0., -1.,
                0., -1., -1., -1.,  1., -1., -1.,  1., -1.,  0.,  0.],
              dtype=float32)
        >>> pf.token_to_segmental_features('a͡ɪ')
        array([ 1.,  1., -1.,  1., -1., -1., -1.,  0.,  1., -1., -1.,  0., -1.,
                0., -1.,  0.,  0., -1., -1., -1.,  0., -1.,  0.,  0.],
              dtype=float32)
        >>> pf.token_to_segmental_features('aɪ')
        array([ 1.,  1., -1.,  1., -1., -1., -1.,  0.,  1., -1., -1.,  0., -1.,
                0., -1.,  0.,  0., -1., -1., -1.,  0., -1.,  0.,  0.],
              dtype=float32)
        """
        vec = self.feature_table.word_to_vector_list(token, numeric=True)
        NUMBER_OF_PANPHON_FEATURES = 24
        # is not IPA
        if not vec:
            return np.zeros(NUMBER_OF_PANPHON_FEATURES, dtype=np.float32)
        # EV and PanPhon tokenization do not match, as with dipthongs
        if len(vec) > 1:
            # TODO: We should warn the user that we are averaging the features due to a mismatch in
            # EV and Panphon tokenization.
            return np.mean(vec, axis=0, dtype=np.float32)
        # EV and PanPhon tokenization matches here
        else:
            return np.array(vec[0], dtype=np.float32)

    def get_features(self, tokens: list[str]) -> npt.NDArray[np.float32]:
        """Get Phonological Feature Vectors by stacking segmental features, tone features, and punctuation features

        Args:
            tokens (list[str]): a list of IPA and normalized punctuation tokens

        Returns:
            npt.NDArray[np.float32]: a thirty-nine-dimensional encoding of segmental (22), tone(7), and punctuation (7) features
        """
        if not tokens:
            return np.array([])
        punctuation_features = self.get_punctuation_features(tokens)
        stress_features = self.get_stress_features(tokens)
        special_token_features = self.get_special_token_features(tokens)
        seg_features = np.vstack([self.token_to_segmental_features(t) for t in tokens])
        assert (
            len(punctuation_features) == len(seg_features) == len(stress_features)
        ), "There should be the same number of segments among segmental, tone, stress, punctuation features"
        return np.concatenate(
            [
                seg_features,
                stress_features,
                punctuation_features,
                special_token_features,
            ],
            axis=1,
            dtype=np.float32,
        )
