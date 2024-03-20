""" EveryVoice performs grapheme-to-phoneme conversion based on language IDs
    All g2p engines must return tokenized characters.
"""
from g2p import get_arpabet_langs, make_g2p
from ipatok import tokenise

AVAILABLE_G2P_ENGINES = {k: "DEFAULT_G2P" for k in get_arpabet_langs()[0]}

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
            text = phonemizer(normalized_input_text).output_string
            return tokenise(text, replace=False, tones=True, unknown=True)

        return g2p_engine
