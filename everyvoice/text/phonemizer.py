""" EveryVoice performs grapheme-to-phoneme conversion based on language IDs
    All g2p engines must return tokenized characters.
"""

import re
from typing import Callable
from unicodedata import normalize

from g2p import get_arpabet_langs, make_g2p
from ipatok import tokenise

AVAILABLE_G2P_ENGINES: dict[str, str | Callable] = {
    k: "DEFAULT_G2P" for k in get_arpabet_langs()[0]
}

# TODO: Add documentation for this
# If you want to override the default g2p engines, do so by the following:
#
# from some_cool_library import some_cool_g2p_method
# AVAILABLE_G2P_ENGINES['YOUR_LANGUAGE_CODE'] = some_cool_g2p_method
#
# IMPORTANT: Your g2p engine must return a list of tokenized symbols, and all of the returned symbols must be defined in your everyvoice-shared-text-config.yaml file.


class CachingG2PEngine:
    """caching tokenizing g2p engine"""

    def __init__(self, lang_id):
        self._cache = {}
        self.phonemizer = make_g2p(lang_id, f"{lang_id}-ipa")

    def process_one_token(self, input_token: str) -> list[str]:
        """Process one input token, dumbly split on whitespace.
        The output can be multiple tokens, since a proper tokenizer is used."""
        # ipatok strips some important characters, so as a hack,
        # we convert them to the private use area first
        PUA_CHARS = ["_", " ", ".", "ˈ", "ˌ"]
        PUA_START_NUMBER = 983040  # U+F0000
        text = self.phonemizer(input_token).output_string
        for i, char in enumerate(PUA_CHARS):
            text = text.replace(char, chr(PUA_START_NUMBER + i))
        tokens = tokenise(text, replace=False, tones=True, strict=False, unknown=True)
        # normalize the output since ipatok applies NFD
        unicode_normalization_form = self.phonemizer.transducers[-1].norm_form.value
        if unicode_normalization_form != "none":
            tokens = [normalize(unicode_normalization_form, token) for token in tokens]
        # convert the pua tokens back to their originals
        for i, token in enumerate(tokens):
            # PUA tokens have length 1
            if len(token) == 1:
                token_ord = ord(token)
                if token_ord >= PUA_START_NUMBER:
                    tokens[i] = PUA_CHARS[token_ord - PUA_START_NUMBER]
        return tokens

    def __call__(self, normalized_input_text: str) -> list[str]:
        input_tokens = re.split(r"(\s+)", normalized_input_text)
        output_tokens = []
        for token in input_tokens:
            cached = self._cache.get(token, None)
            if cached is None:
                cached = self.process_one_token(token)
                self._cache[token] = cached
            output_tokens += cached
        return output_tokens


def get_g2p_engine(lang_id: str):

    if lang_id not in AVAILABLE_G2P_ENGINES:
        raise NotImplementedError(
            f"Sorry, we don't have a grapheme-to-phoneme engine available for {lang_id}."
            " Please follow the docs to implement one yourself, or try training a character-based model instead."
        )

    if AVAILABLE_G2P_ENGINES[lang_id] == "DEFAULT_G2P":
        # Register the engine so we don't have to build it next time
        AVAILABLE_G2P_ENGINES[lang_id] = CachingG2PEngine(lang_id)

    return AVAILABLE_G2P_ENGINES[lang_id]
