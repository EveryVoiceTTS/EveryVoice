from typing import List
from unicodedata import normalize

from panphon import FeatureTable

FT = FeatureTable()


def get_tone_features(text: List[str]) -> List[List[int]]:
    """Return Wang (1967) style tone features.
        - Contour
        - High
        - Central
        - Mid
        - Rising
        - Falling
        - Convex

    *If your language uses phonemic tone you MUST amend this function to match your language
    Panphon does not use these features.*

    Args:
        text (list(str)): segmented phones
    """
    tone_features = []
    high_tone_chars = [
        normalize("NFC", x)
        for x in [
            "áː",
            "á",
            "ʌ̃́ː",
            "ʌ̃́",
            "éː",
            "é",
            "íː",
            "í",
            "ṹː",
            "ṹ",
            "óː",
            "ó",
        ]
    ]
    low_tone_chars = [
        normalize("NFC", x) for x in ["òː", "ũ̀ː", "ìː", "èː", "ʌ̃̀ː", "àː"]
    ]
    for char in text:
        char = normalize("NFC", char)
        if char in high_tone_chars:
            tone_features.append([-1, 1, -1, -1, -1, -1, -1])
        elif char in low_tone_chars:
            tone_features.append([-1, -1, -1, -1, -1, -1, -1])
        else:
            tone_features.append([0, 0, 0, 0, 0, 0, 0])
    return tone_features


def get_punctuation_features(text):
    excl = ["!", "¡"]
    quest = ["?", "¿"]
    silence = ["<SIL>"]
    bb = [".", ":", ";", "…"]
    sb = [","]
    qm = ['"', "'", "“", "”", "«", "»"]
    wb = [" "]
    punctuation_features = []
    for char in text:
        char = normalize("NFC", char)
        if char in excl:
            punctuation_features.append([1, 0, 0, 0, 0, 0, 0])
        elif char in quest:
            punctuation_features.append([0, 1, 0, 0, 0, 0, 0])
        elif char in bb:
            punctuation_features.append([0, 0, 1, 0, 0, 0, 0])
        elif char in sb:
            punctuation_features.append([0, 0, 0, 1, 0, 0, 0])
        elif char in qm:
            punctuation_features.append([0, 0, 0, 0, 1, 0, 0])
        elif char in silence:
            punctuation_features.append([0, 0, 0, 0, 0, 1, 0])
        elif char in wb:
            punctuation_features.append([0, 0, 0, 0, 0, 0, 1])
        else:
            punctuation_features.append([0, 0, 0, 0, 0, 0, 0])
    return punctuation_features


def char_to_vector_list(char):
    vec = FT.word_to_vector_list(char, numeric=True)
    assert len(vec) < 2
    try:
        return vec[0]
    except IndexError:
        breakpoint()


def get_features(tokens):
    """Pass cleaned tokens"""
    # tokenizer = moh_tokenizer if moh else arpa_tokenizer
    # tokens = tokenizer.tokenize(text)
    punctuation_features = get_punctuation_features(tokens)
    tone_features = get_tone_features(tokens)
    spe_features = [char_to_vector_list(t) for t in tokens]
    assert len(punctuation_features) == len(tone_features) == len(spe_features)
    return [
        spe_features[i] + tone_features[i] + punctuation_features[i]
        for i in range(len(spe_features))
    ]
