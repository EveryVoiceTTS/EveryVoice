# This module contains helper functions for use with the configuration wizard CLI
#
#
import json
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union
from unicodedata import normalize

import simple_term_menu
import typer
import yaml
from g2p import make_g2p
from g2p.exceptions import InvalidLanguageCode, NoPath
from g2p.transducer import CompositeTransducer
from loguru import logger
from pydantic import BaseModel
from readalongs.util import get_langs
from rich import print
from rich.panel import Panel
from tqdm import tqdm

from everyvoice.utils import lower, nfc_normalize


def get_single_lang_information(
    supported_langs, unsupported_langs
) -> Tuple[Union[None, Dict[str, str]], Union[None, Dict[str, str]]]:
    """Get language information for the user's dataset through terminal prompts

    Returns:
        Tuple[Union[None, Dict[str, str]], Union[None, Dict[str, str]]]: A tuple containing the selected language or None
    """
    supported_langs_choices = ["[none]: my language isn't here"] + [
        f"[{k}]: {v}" for k, v in supported_langs
    ]
    all_langs_choices = [f"[{lang.alpha_3}]: {lang.name}" for lang in unsupported_langs]
    supported_langs_choice: int = get_menu_prompt(  # type: ignore
        "Which of the following supported languages are in your dataset?",
        supported_langs_choices,
        multi=False,
        search=True,
    )
    unsupported_langs_choice = None
    if supported_langs_choice == 0:
        unsupported_langs_choice: int = get_menu_prompt(  # type: ignore
            "Please select all the languages in your dataset:",
            all_langs_choices,
            multi=False,
            search=True,
        )
    return (
        supported_langs[supported_langs_choice] if supported_langs_choice else None,
        unsupported_langs[unsupported_langs_choice]
        if unsupported_langs_choice
        else None,
    )


def get_symbols_from_g2p_mapping(in_lang: str, out_lang: str) -> Dict[str, List[str]]:
    try:
        transducer = make_g2p(in_lang, out_lang)
    except (InvalidLanguageCode, NoPath) as e:
        logger.warning(e)
        return {}
    if isinstance(transducer, CompositeTransducer):
        chars = transducer._transducers[-1].mapping.mapping
    else:
        chars = transducer.mapping.mapping
    return {
        f"{in_lang}_ipa": list({normalize("NFC", c["out"]) for c in chars}),
        f"{in_lang}_char": list({normalize("NFC", c["in"]) for c in chars}),
    }


def get_lang_information() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Get language information for the user's dataset through terminal prompts

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]: A tuple containing Supported Languages and Unsupported Languages - each are lists of dicts with iso code keys and language name values
    """
    logger.info("Getting supported languages...")
    from pycountry import languages

    supported_langs = get_langs()[1].items()
    all_langs = list(languages)
    supported_langs_choices = ["[none]: my language isn't here"] + [
        f"[{k}]: {v}" for k, v in supported_langs
    ]
    all_langs_choices = [f"[{lang.alpha_3}]: {lang.name}" for lang in languages]
    langs: List[int] = get_menu_prompt(  # type: ignore
        "Which of the following supported languages are in your dataset?",
        supported_langs_choices,
        multi=True,
        search=True,
    )
    unsupported_langs: List[int] = []
    if 0 in langs:
        unsupported_langs: List[int] = get_menu_prompt(  # type: ignore
            "Please select all the languages in your dataset:",
            all_langs_choices,
            multi=True,
            search=True,
        )
    langs = [x - 1 for x in langs if x]  # remove "none" index from count
    supported_langs = list(supported_langs)
    return {supported_langs[i][0]: supported_langs[i][1] for i in langs}, {
        all_langs[i].alpha_3: all_langs[i].name for i in unsupported_langs
    }


def get_symbol_set(
    filelist_data: List[Dict[str, str]],
    supported_langs,
    unsupported_langs,
    to_filter="-';:,.!?¡¿—…\"«»“” ",
):
    symbols = {}
    for lang in supported_langs.keys():
        if lang == "eng":
            symbols["eng"] = list(string.ascii_letters)
        else:
            symbols = {**symbols, **get_symbols_from_g2p_mapping(lang, f"{lang}-ipa")}
    for iso, lang in unsupported_langs.items():
        logger.info(
            f"We don't support [{iso}] - {lang} so we will just guess. You can always update/change the symbol set in your configuration file later."
        )
        possible_values = [iso, lang, "default"]
        possible_text_transformations = ["none", lower, nfc_normalize]
        text_transformations: List[int] = get_menu_prompt(  # type: ignore
            f"Please select all the text transformations to apply to [{iso}] - {lang} before determining symbol set:",
            [
                "None",
                "Lowercase",
                "NFC Normalization - See here for more information: https://withblue.ink/2019/03/11/why-you-need-to-normalize-unicode-strings.html",
            ],
            multi=True,
        )
        chars = set()
        for row in tqdm(filelist_data):
            row_lang = row["language"]
            if row_lang in possible_values:
                text = row["text"]
                if 0 not in text_transformations:
                    for t in text_transformations:
                        text = possible_text_transformations[t](text)
                for c in text:
                    chars.add(c)
        symbols[iso] = [
            x for x in chars if x not in to_filter
        ]  # filter default punctuation
    return {k: sorted(v) for k, v in symbols.items()}


def auto_check_audio(
    filelist_data: List[Dict[str, str]], wavs_dir: Path
) -> Tuple[int, float, float]:
    """Check all audio with basenames from filelist in wavs_dir and return the sampling rate,
    min length rounded down (seconds), and max length rounded up (seconds). Log warnings if less than
    0.25 seconds or longer that 11 seconds.
    """
    import math

    srs_counter: Counter[int] = Counter()
    logger.info("🧙 is checking your audio. please wait...")
    lengths = []
    from scipy import signal
    from scipy.io import wavfile

    double_check = typer.confirm(
        "The configuration wizard can double check that your declared sampling rate is correct by analyzing the spectrogram. Would you like to do this? It will mean that your processing takes longer."
    )
    for d in tqdm(filelist_data):
        audio_path = wavs_dir / (d["basename"] + ".wav")
        try:
            samplerate, data = wavfile.read(audio_path)
        except FileNotFoundError:
            logger.warning(
                f"File '{audio_path}' was not found. Please ensure the file exists and your filelist is created properly."
            )
            continue
        if double_check:
            frequencies, _, spectrogram = signal.spectrogram(data, samplerate)
            energy_threshold = None
            n_empty_bands = 0
            for i, band in enumerate(spectrogram):
                if sum(band) < 0.1:
                    if n_empty_bands == 0:
                        energy_threshold = int(frequencies[i])
                    n_empty_bands += 1
                    if n_empty_bands > 4:
                        logger.warning(
                            f"File '{audio_path} was labelled as having a sampling rate of {samplerate}, but it appears to lose energy at around {energy_threshold}Hz. We'll skip this file."
                        )
        srs_counter[samplerate] += 1
        lengths.append(data.shape[0] / samplerate)
    for k, v in srs_counter.items():
        logger.info(f"{v} audio files found at {k}Hz sampling rate")
    if not srs_counter:
        logger.error(f"Couldn't read any files at {wavs_dir}. Please check your path.")
        exit()
    sr = min(srs_counter.keys())
    logger.info(f"Using {sr}Hz sampling rate")
    min_s = min(lengths)
    if min_s < 0.25:
        logger.warning(
            f"Shortest sample was {min_s} seconds long - this is probably too short. Please remove all audio shorter than 0.25 seconds."
        )
    else:
        logger.info(f"Shortest sample was {min_s} seconds long")
    max_s = max(lengths)
    if max_s > 11:
        logger.warning(
            f"Longest sample was {max_s} seconds long - this is probably too long, and may result in longer training times or force you to use a reduced batch size. Please remove all audio longer than 11 seconds."
        )
    else:
        logger.info(f"Longest sample was {max_s} seconds long")
    return sr, float(math.floor(min_s)), float(math.ceil(max_s))


def get_menu_prompt(
    prompt_text: str, choices: List[str], multi=False, search=False
) -> Union[int, List[int]]:
    """Given some prompt text and a list of choices, create a simple terminal window
       and return the index of the choice

    Args:
        prompt_text (str): rich prompt text to print before menu
        choices (List[str]): choices to display

    Returns:
        int: index of choice
    """
    print(Panel(prompt_text))
    menu = simple_term_menu.TerminalMenu(
        choices,
        multi_select=multi,
        show_multi_select_hint=multi,
        show_search_hint=search,
    )
    index = menu.show()
    sys.stdout.write("\033[K")
    if index is None:
        exit()
    return index


def get_required_headers(headers: List[str], sample_data: List[List[str]]) -> List[str]:
    """Given some headers and sample data, prompt the user to specify which column is text and which column is basenames
       the basename should be unique for each audio file and should be without a file extension.

    Args:
        headers (List[str]): hypothesized list of headers
        sample_data (List[List[str]]): sample data from reading in filelist in psv format (like LJSpeech metadata file)

    Returns:
        List[str]: revised list of headers
    """
    text_column: int = get_menu_prompt(  # type: ignore
        "These are the values from the first row in your data. Which column contains the [bold blue]text?",
        [str(f"{x}: {sample_data[0][x]}") for x in range(len(headers))],
    )
    bn_column: int = get_menu_prompt(  # type: ignore
        "Which column contains the [bold blue]basenames?",
        [
            str(f"{x}: {sample_data[0][x]}")
            for x in range(len(headers))
            if str(x) != text_column
        ],
    )
    headers[text_column] = "text"
    headers[bn_column] = "basename"
    return headers


def create_default_filelist(
    data: List[List[str]], headers: List[str]
) -> List[Dict[str, str]]:
    """Given a list of row data (list of cell strings), and a list of headers, produce a list
        of dicts that could be read by csv.DictReader and written by csv.DictWriter

    Args:
        data (List[List[str]]): spreadsheet row data
        headers (List[str]): list of headers

    Returns:
        List[Dict[str, str]]: List of spreadsheet data in dict format
    """
    len_data = len(data[0])
    new_data = []
    for d in data:
        data_dict = {
            h: d[h_i] if h_i < len_data else "default" for h_i, h in enumerate(headers)
        }
        new_data.append(data_dict)
    return new_data


def write_config_to_file(config: BaseModel, path: Path):
    """Given a Pydantic model, convert it to a json object and write as yaml or json

    Args:
        config (BaseModel): The Configuration to write
        path (Path): The output path; must end with either json or yaml
    """
    config = json.loads(config.json())
    with open(path, "w", encoding="utf8") as f:
        path_string = str(path)
        if path_string.endswith("yaml"):
            yaml.dump(config, f, default_flow_style=None, allow_unicode=True)
        else:
            json.dump(config, f, ensure_ascii=False)


def write_dict_to_config(config: Dict, path: Path):
    """Given an object, write it to file

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


def create_sox_effects_list(indices: List[str], sr: int):
    effects = [["channels", "1"]]
    if "none" in indices:
        return effects
    if "resample" in indices:
        effects.append(["rate", str(sr)])
    if "norm" in indices:
        effects.append(["norm", "-3.0"])
    if "sil-start" in indices:
        effects.append(["silence", "1", "0.1", "1.0%"])
    return effects
