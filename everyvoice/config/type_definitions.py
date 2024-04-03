"""
This file is for light-weight type definitions with no dependencies.
The file is intended to load in a few milliseconds.
More expensive type definitions belong in shared_types.py.
"""

from enum import Enum


class DatasetTextRepresentation(str, Enum):
    characters = "characters"
    ipa_phones = "phones"
    arpabet = "arpabet"  # always gets mapped to phones


class TargetTrainingTextRepresentationLevel(str, Enum):
    characters = "characters"
    ipa_phones = "phones"
    phonological_features = "phonological_features"
