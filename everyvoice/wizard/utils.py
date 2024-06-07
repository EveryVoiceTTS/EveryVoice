import csv
import json
from collections import UserDict
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable

import yaml
from tqdm import tqdm

from everyvoice.config.type_definitions import (
    DatasetTextRepresentation,
    TargetTrainingTextRepresentationLevel,
)


def rename_unknown_headers(headers):
    for i, header in enumerate(headers):
        if header not in ["basename", "raw_text", "speaker", "language"] + [
            x.value for x in DatasetTextRepresentation
        ]:
            headers[i] = f"unknown_{i}"
    return headers


def apply_automatic_text_conversions(
    filelist_data, text_representation, global_isocode=None
) -> str:
    """Arpabet is automatically converted to phones. phones are automatically derived from characters
       if a corresponding g2p module is available.

    Args:
        filelist_data (list[dict]): _description_
        text_representation (DatasetTextRepresentation): _description_
        isocode (str, optional): _description_. Defaults to None.
    Returns:
        target_training_representation (str): the target training representation level. returns 'phones' unless there is more data available from using 'characters'.
    """
    from everyvoice.text.arpabet import ARPABET_TO_IPA_TRANSDUCER
    from everyvoice.text.phonemizer import AVAILABLE_G2P_ENGINES, get_g2p_engine

    g2p_engines = {}
    character_counter = 0
    phone_counter = 0
    for item in tqdm(filelist_data, desc=f"Processing your {text_representation}"):
        # add globally defined language code
        if global_isocode is not None:
            item["language"] = global_isocode
            item_isocode = global_isocode
        else:
            # define the isocode from the item
            item_isocode = item.get("language", None)
        # convert arpabet to phones
        if (
            DatasetTextRepresentation.arpabet.value in item
            and DatasetTextRepresentation.ipa_phones.value not in item
        ):
            item[DatasetTextRepresentation.ipa_phones.value] = (
                ARPABET_TO_IPA_TRANSDUCER(
                    item[DatasetTextRepresentation.arpabet.value]
                ).output_string
            )
        # if phones don't exist but g2p is available, calculate them
        if (
            DatasetTextRepresentation.characters.value in item
            and DatasetTextRepresentation.ipa_phones.value not in item
            and item_isocode is not None
        ):
            if item_isocode in g2p_engines:
                pass  # just use the pre-loaded g2p engine.
            elif item_isocode in AVAILABLE_G2P_ENGINES:
                g2p_engines[item_isocode] = get_g2p_engine(item_isocode)
            else:
                g2p_engines[item_isocode] = None
            g2p_engine = g2p_engines[item_isocode]
            if g2p_engine is not None:
                phone_string_tokens = g2p_engine(
                    item[DatasetTextRepresentation.characters.value]
                )
                item[DatasetTextRepresentation.ipa_phones.value] = "".join(
                    phone_string_tokens
                )
        if DatasetTextRepresentation.ipa_phones.value in item:
            phone_counter += 1
        if DatasetTextRepresentation.characters.value in item:
            character_counter += 1
    if character_counter > phone_counter:
        target_training_representation = (
            TargetTrainingTextRepresentationLevel.characters.value
        )
    else:
        target_training_representation = (
            TargetTrainingTextRepresentationLevel.ipa_phones.value
        )
    if g2p_engines:
        for engine_iso, engine_value in g2p_engines.items():
            if engine_value is None:
                print(
                    f"Your text data contains characters and the language id {engine_iso} is not supported by a grapheme-to-phoneme engine. If you want to train using a pronunciation form for {engine_iso} (which usually results in better quality) you will have to add a g2p engine for {engine_iso}. Please see <TODO:Docs> for more information."
                )
    return target_training_representation


def read_unknown_tabular_filelist(
    path,
    delimiter=",",
    quoting=csv.QUOTE_NONE,
    escapechar="\\",
    record_limit: int = 0,  # if non-zero, read only this many records
) -> list[list[str]]:
    """Returns a list of list of cell values instead of a list of dicts as with
        the standard everyvoice.utils.generic_*sv_filelist_reader

    Returns:
        list[list[str]]: a list of rows containing a list of cell values
    """
    f: Iterable[str]
    with open(path, "r", newline="", encoding="utf8") as f:
        if record_limit:
            f = islice(f, record_limit)
        reader = csv.reader(
            f,
            delimiter=delimiter,
            quoting=quoting,
            escapechar=escapechar,
        )
        files = list(reader)
    return files


def write_dict_to_config(config: Dict, path: Path):
    """Given an object, write it to file.
       We have to serialize the json first to make use of the custom serializers,
       and we don't always write the json directly since we might want to add extra
       information after serialization.

    Args:
        config (Dict): The Configuration Dict to write
        path (Path): The output path; must end with either json or yaml
    """
    with open(path, "w", encoding="utf8") as f:
        path_string = str(path)
        if path_string.endswith("yaml"):
            yaml.dump(config, f, default_flow_style=None, allow_unicode=True)
        else:
            json.dump(config, f, ensure_ascii=False)


class EnumDict(UserDict):
    """dict that accepts Enum elements to mean their string values"""

    def convert_key(self, key):
        if isinstance(key, Enum):
            return key.value
        else:
            return key

    def __getitem__(self, key):
        return super().__getitem__(self.convert_key(key))

    def __setitem__(self, key, value):
        return super().__setitem__(self.convert_key(key), value)
