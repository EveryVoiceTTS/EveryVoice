""" EveryVoice performs grapheme-to-phoneme conversion based on language IDs
    All g2p engines must return tokenized characters.
"""
from typing import Callable

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


def get_g2p_engine(lang_id: str):

    if lang_id not in AVAILABLE_G2P_ENGINES:
        raise NotImplementedError(
            f"Sorry, we don't have a grapheme-to-phoneme engine available for {lang_id}. Please follow the docs to implement one yourself, or try training a character-based model instead."
        )

    if AVAILABLE_G2P_ENGINES[lang_id] == "DEFAULT_G2P":
        phonemizer = make_g2p(lang_id, f"{lang_id}-ipa")

        def g2p_engine(normalized_input_text: str) -> list[str]:
            # ipatok strips some important characters, so as a hack, we convert them to the private use area first
            PUA_CHARS = ["_", " ", ".", "ˈ", "ˌ"]
            text = phonemizer(normalized_input_text).output_string
            for i, char in enumerate(PUA_CHARS):
                text = text.replace(char, chr(983040 + i))
            tokens = tokenise(
                text, replace=False, tones=True, strict=False, unknown=True
            )
            # convert the pua tokens back to their originals
            for i, token in enumerate(tokens):
                if len(token) == 1:
                    token_ord = ord(token)
                    if token_ord >= 983040:
                        tokens[i] = PUA_CHARS[token_ord - 983040]
            return tokens

        # Register the engine so we don't have to build it next time
        AVAILABLE_G2P_ENGINES[lang_id] = g2p_engine

    return AVAILABLE_G2P_ENGINES[lang_id]
