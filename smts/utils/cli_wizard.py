# This module contains helper functions for use with the configuration wizard CLI
#
#
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

import simple_term_menu
import yaml
from loguru import logger
from pydantic import BaseModel
from rich import print
from rich.panel import Panel
from tqdm import tqdm


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
    from scipy.io import wavfile

    for d in tqdm(filelist_data):
        audio_path = wavs_dir / (d["basename"] + ".wav")
        try:
            samplerate, data = wavfile.read(audio_path)
        except FileNotFoundError:
            logger.warning(
                f"File '{audio_path}' was not found. Please ensure the file exists and your filelist is created properly."
            )
        srs_counter[samplerate] += 1
        lengths.append(data.shape[0] / samplerate)
    for k, v in srs_counter.items():
        logger.info(f"{v} audio files found at {k}Hz sampling rate")
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
            f"Longest sample was {min_s} seconds long - this is probably too long, and may result in longer training times or force you to use a reduced batch size. Please remove all audio longer than 11 seconds."
        )
    else:
        logger.info(f"Longest sample was {max_s} seconds long")
    return sr, float(math.floor(min_s)), float(math.ceil(max_s))


def get_menu_prompt(
    prompt_text: str, choices: List[str], multi=False
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
        choices, multi_select=multi, show_multi_select_hint=multi
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
