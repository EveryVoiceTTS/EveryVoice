import glob
import os
import random
import sys
from copy import copy, deepcopy
from pathlib import Path
from typing import Sequence

import questionary
from rich import print as rich_print
from rich.panel import Panel
from rich.style import Style
from tqdm import tqdm

from everyvoice.config.type_definitions import DatasetTextRepresentation
from everyvoice.text.utils import guess_graphemes_in_text, guess_ipa_phones_in_text
from everyvoice.utils import lower, nfc_normalize, read_festival, slugify
from everyvoice.wizard import TEXT_CONFIG_FILENAME_PREFIX, Step, StepNames, Tour
from everyvoice.wizard.prompts import (
    CUSTOM_QUESTIONARY_STYLE,
    get_response_from_menu_prompt,
)
from everyvoice.wizard.utils import (
    apply_automatic_text_conversions,
    get_iso_code,
    has_columns_left,
    read_unknown_tabular_filelist,
    rename_unknown_headers,
    sanitize_paths,
)
from everyvoice.wizard.validators import validate_path

# WAVS & FILELIST


class DatasetNameStep(Step):
    DEFAULT_NAME = StepNames.dataset_name_step
    REVERSIBLE = True

    def prompt(self):
        return input(
            "What would you like to call this dataset? This is needed because EveryVoice lets you train models with multiple sources of data. Please choose a name that distinguishes this data source, e.g. 'john-english' or 'maria-spanish' or something similarly descriptive: "
        )

    def validate(self, response):
        if len(response) == 0:
            rich_print("Sorry, your dataset needs a name.")
            return False
        slug = slugify(response)
        if not slug == response:
            rich_print(
                f"Sorry, your name: '{response}' is not valid, since it will be used to create a file and special characters are not permitted in filenames. Please re-type something like {slug} instead."
            )
            return False
        return True

    def effect(self):
        rich_print(
            f"Great! The Configuration Wizard 🧙 finished the configuration for your dataset named '{self.response}'"
        )


class DatasetPermissionStep(Step):
    DEFAULT_NAME = StepNames.dataset_permission_step
    REVERSIBLE = True
    choices = (
        "No, I don't have permission to use this data.",
        "Yes, I do have permission to use this data.",
    )

    def prompt(self):
        prompt_text = """Do you have permission to use this data to build a TTS model? It is unethical to build a TTS model of a speaker without their knowledge or permission and there can be serious consequences for doing so."""
        return get_response_from_menu_prompt(
            prompt_text=prompt_text,
            choices=self.choices,
        )

    def validate(self, response):
        return response in self.choices

    def effect(self):
        if self.response.startswith("No"):
            rich_print("OK, we'll ask you to choose another dataset then!")
            self.children = ()
            self.tour.remove_dataset(self.state_subset)

            # Permission can be revoked, but if you said no you can't go back because
            # we destroy too much of the state here. Just pick another data file.
            self.REVERSIBLE = False

    def undo(self):
        # Do not call super().undo() here! We don't want to remove the dataset steps
        self.response = None
        self.completed = False
        self.state.pop(self.name, None)


class WavsDirStep(Step):
    DEFAULT_NAME = StepNames.wavs_dir_step
    REVERSIBLE = True

    def prompt(self):
        return questionary.path(
            "Where are your audio files?",
            style=CUSTOM_QUESTIONARY_STYLE,
            only_directories=True,
        ).unsafe_ask()

    def sanitize_input(self, response):
        return sanitize_paths(response)

    def validate(self, response) -> bool:
        valid_path = validate_path(response, is_dir=True, exists=True)
        if not valid_path:
            return False
        path_expanded = Path(response).expanduser()
        glob_iter = glob.iglob(os.path.join(path_expanded, "**/*.wav"), recursive=True)
        contains_wavs = next(glob_iter, None) is not None
        if not contains_wavs:
            rich_print(
                f"Sorry, no .wav files were found in '{path_expanded}'. Please choose a directory with audio files."
            )
        return valid_path and contains_wavs


class SampleRateConfigStep(Step):
    DEFAULT_NAME = StepNames.sample_rate_config_step
    REVERSIBLE = True

    def prompt(self):
        return questionary.text(
            "What is the sample rate (in Hertz) of your data?",
            style=CUSTOM_QUESTIONARY_STYLE,
        ).unsafe_ask()

    def validate(self, response):
        try:
            self.response = int(response)
            if self.response < 100 or float(response) != self.response:
                rich_print(
                    f"{response} is not a valid sample rate. Please enter an integer representing the sample rate in Hertz of your data."
                )
                return False
            return True
        except ValueError:
            rich_print(
                f"{response} is not a valid sample rate. Please enter an integer representing the sample rate in Hertz of your data."
            )
            return False


class FilelistStep(Step):
    DEFAULT_NAME = StepNames.filelist_step
    REVERSIBLE = True

    def prompt(self):
        return questionary.path(
            "Where is your data filelist?", style=CUSTOM_QUESTIONARY_STYLE
        ).unsafe_ask()

    def sanitize_input(self, response):
        return sanitize_paths(response)

    def validate(self, response) -> bool:
        return validate_path(response, is_file=True, exists=True)


class FilelistFormatStep(Step):
    DEFAULT_NAME = StepNames.filelist_format_step
    REVERSIBLE = True
    separators = {"psv": "|", "tsv": "\t", "csv": ","}

    def prompt(self):
        return get_response_from_menu_prompt(
            prompt_text="Select which format your filelist is in:",
            choices=("psv", "tsv", "csv", "festival"),
        )

    def looks_like_sv(self, file_type, separator) -> bool:
        assert self.state
        filelist_path = self.state.get(StepNames.filelist_step)
        initial_records = read_unknown_tabular_filelist(
            filelist_path, delimiter=separator, record_limit=10
        )

        if len(initial_records) > 0:
            column_count = len(initial_records[0])
        else:
            rich_print(f"ERROR: File ({filelist_path} is empty. Please double check.")
            sys.exit(1)

        if column_count < 2:
            rich_print(
                f"File '{filelist_path}' does not look like a '{file_type}' file: no record separator found on header line."
            )
            return False

        for i, record in enumerate(initial_records):
            if len(record) != column_count:
                rich_print(
                    f"File '{filelist_path}' does not look like a '{file_type}' file: the {i}th record has a different number of fields than the header row."
                )
                return False

        return True

    def validate(self, response):
        if response == "festival":
            filelist_path = self.state.get(StepNames.filelist_step)
            try:
                _ = read_festival(filelist_path, 10)
                return True
            except ValueError:
                rich_print(f"File '{filelist_path}' is not in the festival format.")
                return False

        separator = self.separators.get(response, None)
        if separator:
            return self.looks_like_sv(response, separator)

        assert False and "the above code covers all valid cases"  # pragma: no cover

    def effect(self):
        """
        This effect occurs after the filelist format is selected. The Tour.state should now
        have both the filelist path and the filelist data type, so we parse the data and add
        the data to the tour state. We then inspect the headers for the data and add steps to
        select the basename and text if they are not already specified in the headers.
        """
        self.saved_state = {
            "filelist_delimiter": None,
            "filelist_headers": None,
            "filelist_data_list": None,
            "filelist_data": None,
            "selected_headers": None,
        }

        file_type = self.state.get(StepNames.filelist_format_step)
        filelist_path = self.state.get(StepNames.filelist_step)
        if not isinstance(filelist_path, Path):
            filelist_path = Path(filelist_path).expanduser()
        if file_type == "csv":
            self.state["filelist_delimiter"] = ","
        elif file_type == "psv":
            self.state["filelist_delimiter"] = "|"
        elif file_type == "tsv":
            self.state["filelist_delimiter"] = "\t"
        else:
            self.state["filelist_delimiter"] = None
        if file_type == "festival":
            filelist_data_dict = read_festival(filelist_path)
            self.state["filelist_headers"] = list(filelist_data_dict[0].keys())
            self.state["filelist_data"] = filelist_data_dict
        else:
            self.state["filelist_data_list"] = read_unknown_tabular_filelist(
                filelist_path, delimiter=self.state.get("filelist_delimiter")
            )
            self.state["filelist_headers"] = list(self.state["filelist_data_list"][0])
            standard_header_found = False
            self.state["selected_headers"] = []
            try:
                text_index = self.state["filelist_headers"].index(
                    "text"
                )  # call it text until FilelistTextRepresentationStep
                self.state["selected_headers"].append(text_index)
                standard_header_found = True
            except ValueError:
                self.tour.add_step(
                    HeaderStep(
                        name=StepNames.text_header_step,
                        prompt_text="Which column contains the [bold blue]text?",
                        header_name="text",
                        state_subset=self.state_subset,
                    ),
                    self,
                )
            try:
                basename_index = self.state["filelist_headers"].index("basename")
                self.state["selected_headers"].append(basename_index)
                standard_header_found = True
            except ValueError:
                self.tour.add_step(
                    HeaderStep(
                        name=StepNames.basename_header_step,
                        prompt_text="Which column contains the [bold blue]basenames?",
                        header_name="basename",
                        state_subset=self.state_subset,
                    ),
                    self,
                )
            if not standard_header_found:
                self.tour.add_step(
                    HasHeaderLineStep(state_subset=self.state_subset), self
                )


class ValidateWavsStep(Step):
    DEFAULT_NAME = StepNames.validate_wavs_step
    AUTOMATIC = True
    REVERSIBLE = True

    def wav_file_early_validation(self) -> int:
        """Look for missing wav files and return the error count"""
        assert self.state is not None  # fixes mypy errors
        wavs_dir = Path(self.state[StepNames.wavs_dir_step])
        files_not_found = []
        MAX_SAMPLES = 1000
        filelist_data = self.state["filelist_data"]
        file_list_size = len(filelist_data)
        sample: Sequence[int]
        if file_list_size > MAX_SAMPLES:
            rich_print(
                f"Checking a sample of {MAX_SAMPLES} of your audio files to make sure they are present."
            )
            sampled_text = " sampled"
            sample = sorted(random.sample(range(file_list_size), MAX_SAMPLES))
        else:
            rich_print("Checking if all your audio files are present.")
            sampled_text = ""
            sample = range(file_list_size)
        for item in sample:
            record = filelist_data[item]  # +1 to skip past header
            wav_basename = record["basename"]
            wav_filename = wavs_dir / (
                wav_basename + ("" if wav_basename.endswith(".wav") else ".wav")
            )
            if not wav_filename.exists():
                files_not_found.append(wav_filename)
        if files_not_found:
            n = len(files_not_found)
            if n == 1:
                rich_print(
                    f"Warning: wav file '{files_not_found[0]}' was not found, please check your filelist."
                )
            else:
                rich_print(
                    f"Warning: {n}{sampled_text} wav files were not found, including '{files_not_found[0]}' and '{files_not_found[1]}'.\nPlease check your wavs directory '{wavs_dir}' and your filelist."
                )
            return n
        rich_print(
            f"Great! All{sampled_text} audio files found in directory '{wavs_dir}'."
        )
        return 0

    def prompt(self):
        error_count = self.wav_file_early_validation()
        if error_count:
            return get_response_from_menu_prompt(
                title="Do you want to pick a different wavs directory?",
                choices=(
                    "Yes",
                    "No, I will fix my audio basenames or add missing audio files later.",
                ),
            )
        else:
            return "OK"

    def validate(self, response):
        return response[:3] in ("OK", "Yes", "No,")

    def effect(self):
        if self.response == "Yes":
            self.tour.add_steps(
                [
                    WavsDirStep(state_subset=self.state_subset),
                    ValidateWavsStep(state_subset=self.state_subset),
                ],
                self,
            )
        elif self.response.startswith("No"):
            rich_print(
                Panel(
                    "Continuing despite missing audio files. Make sure you fix your filelist later or add missing audio files, otherwise entries in your filelist with missing audio files will be skipped during preprocessing and therefore be ignored during training.",
                    title="Missing audio files",
                    border_style=Style(color="#EF1010"),
                )
            )


class FilelistTextRepresentationStep(Step):
    DEFAULT_NAME = StepNames.filelist_text_representation_step
    REVERSIBLE = True
    text_representation_options = tuple(x.value for x in DatasetTextRepresentation)

    def prompt(self):
        return get_response_from_menu_prompt(
            prompt_text=f"Which representation is your text in? Choose '{DatasetTextRepresentation.ipa_phones.value}' if your text data only uses International Phonetic Alphabet characters (punctuation is also OK). Choose '{DatasetTextRepresentation.arpabet}' if your text data uses all ARPABET (punctuation is OK). Choose '{DatasetTextRepresentation.characters}' otherwise.",
            choices=self.text_representation_options,
        )

    def validate(self, response):
        return response in self.text_representation_options

    def effect(self):
        # Apply the text representation level as the new alias for text:
        self.saved_state = {
            "filelist_headers": copy(self.state["filelist_headers"]),
        }
        for i, header in enumerate(self.state["filelist_headers"]):
            if header == "text":
                self.state["filelist_headers"][i] = self.response
        # Rename the "text" key according to the new alias:
        if "filelist_data" in self.state:
            self.saved_state["filelist_data"] = deepcopy(self.state["filelist_data"])
            for item in self.state["filelist_data"]:
                filelist_format = self.state[
                    StepNames.filelist_text_representation_step
                ]
                item[filelist_format] = item.pop("text")


# HEADER SELECTION


class HeaderStep(Step):
    DEFAULT_NAME = StepNames.text_header_step
    REVERSIBLE = True

    def __init__(self, name: str, prompt_text: str, header_name: str, **kwargs):
        super(HeaderStep, self).__init__(name=name, **kwargs)
        self.prompt_text = prompt_text
        self.header_name = header_name

    def prompt(self):
        selected_headers = self.state.get("selected_headers", [])
        choice_indices = [
            x
            for x in range(len(self.state["filelist_headers"]))
            if x not in selected_headers
        ]
        choices = [
            f"{x}: {self.state['filelist_data_list'][0][x]}" for x in choice_indices
        ]
        response = get_response_from_menu_prompt(
            prompt_text=self.prompt_text,
            choices=choices,
            return_indices=True,
        )
        return choice_indices[response]

    def validate(self, response):
        return isinstance(response, int)

    def effect(self):
        # Rename the filelist header with the standard header name
        if "selected_headers" not in self.state:
            self.state["selected_headers"] = []
        self.saved_state = {
            "selected_headers": copy(self.state["selected_headers"]),
            "filelist_headers": copy(self.state["filelist_headers"]),
        }
        self.state["selected_headers"].append(self.response)
        self.state["filelist_headers"][self.response] = self.header_name


class LanguageHeaderStep(HeaderStep):
    REVERSIBLE = True

    def effect(self):
        self.saved_state = {
            "filelist_headers": copy(self.state["filelist_headers"]),
            "selected_headers": copy(self.state.get("selected_headers", None)),
            "filelist_data": deepcopy(self.state.get("filelist_data", None)),
            "model_target_training_text_representation": None,
        }
        # Rename the filelist header with the standard header name
        if "selected_headers" not in self.state:
            self.state["selected_headers"] = []
        self.state["selected_headers"].append(self.response)
        self.state["filelist_headers"][self.response] = self.header_name
        # Rename unselected headers to unknown:
        self.state["filelist_headers"] = rename_unknown_headers(
            self.state["filelist_headers"]
        )
        # re-parse data:
        reload_filelist_data_as_dict(self.state)
        # Add speaker IDs if they are not specified in the filelist
        if self.state[StepNames.data_has_speaker_value_step] == "no":
            add_missing_speaker(self.state)
        # apply automatic conversions
        self.state["model_target_training_text_representation"] = (
            apply_automatic_text_conversions(
                self.state["filelist_data"],
                self.state[StepNames.filelist_text_representation_step],
            )
        )


class HasHeaderLineStep(Step):
    """Check if the data set has a header line, and insert one if not.

    Subsequent steps can systematically assume that self.state["filelist_data_list"][0]
    is the header row."""

    DEFAULT_NAME = StepNames.data_has_header_line_step
    REVERSIBLE = True
    choices = ("no", "yes")

    def prompt(self):
        prompt_text = (
            "Your filelist does not have the standard headers. The first row is:\n"
            + self.state["filelist_delimiter"].join(self.state["filelist_data_list"][0])
            + "\nIs this line a header row?"
        )
        return get_response_from_menu_prompt(
            prompt_text=prompt_text,
            choices=self.choices,
        )

    def validate(self, response):
        return response in self.choices

    def effect(self):
        if self.response == "no":
            rich_print("Reinterpreting your first row as a record, not headers.")
            self.state["filelist_data_list"].insert(
                0, self.state["filelist_data_list"][0]
            )

    def undo(self):
        if self.response == "no":
            self.state["filelist_data_list"].pop(0)
        return super().undo()


class HasSpeakerStep(Step):
    DEFAULT_NAME = StepNames.data_has_speaker_value_step
    REVERSIBLE = True
    choices = ("no", "yes")

    def prompt(self):
        if not has_columns_left(self.state):
            rich_print("No columns available to have a speaker column.")
            return "no"
        else:
            return get_response_from_menu_prompt(
                prompt_text="Does your data have a column/value for the speaker?",
                choices=self.choices,
            )

    def validate(self, response):
        return response in self.choices

    def effect(self):
        self.AUTOMATIC = not has_columns_left(self.state)
        rich_print(
            "Note: if your dataset has speakers with names matching with speakers from other provided datasets, they will be considered the same. If this is not the desired behaviour, you will have to alter the speaker IDs in the relevant datasets to indicate that they are different."
        )
        if self.state[StepNames.data_has_speaker_value_step] == "yes":
            self.tour.add_step(
                HeaderStep(
                    name=StepNames.speaker_header_step,
                    prompt_text="These are the remaining values from the first row in your data. Which column contains the [bold blue]speaker?",
                    header_name="speaker",
                    state_subset=self.state_subset,
                ),
                self,
            )
        else:
            self.tour.add_step(KnowSpeakerStep(state_subset=self.state_subset), self)


class KnowSpeakerStep(Step):
    DEFAULT_NAME = StepNames.know_speaker_step
    REVERSIBLE = True
    choices = ("no", "yes")

    @property
    def dataset_index(self) -> str:
        return self.state_subset.split("_")[-1]

    def prompt(self):
        return get_response_from_menu_prompt(
            choices=self.choices,
            title=f"Since your data does not have a speaker column, we will use a default ID of 'speaker_{self.dataset_index}'. Would you like to specify an alternative speaker ID for this dataset instead?",
        )

    def validate(self, response):
        return response in self.choices

    def effect(self):
        if self.state[StepNames.know_speaker_step] == "yes":
            self.tour.add_step(AddSpeakerStep(state_subset=self.state_subset), self)
        else:
            # Even though AddSpeakerStep is not run, the speaker ID is assigned to its keyword
            self.state[StepNames.add_speaker_step] = f"speaker_{self.dataset_index}"
            rich_print(
                f"OK, '{self.state[StepNames.add_speaker_step]}' will be used as a speaker ID in this dataset then."
            )


class AddSpeakerStep(Step):
    DEFAULT_NAME = StepNames.add_speaker_step
    REVERSIBLE = True

    def prompt(self):
        return input("Please enter the desired speaker ID: ")

    def validate(self, response):
        if len(response) == 0:
            rich_print("Sorry, the speaker needs an ID.")
            return False
        slug = slugify(response)
        if not slug == response:
            rich_print(
                f"Sorry, your ID: '{response}' is not valid. Please avoid using special characters in it and re-type something like {slug} instead."
            )
            return False
        return True

    def effect(self):
        rich_print(
            f"Great! '{self.response}' will be used as the speaker ID for this dataset."
        )


class HasLanguageStep(Step):
    DEFAULT_NAME = StepNames.data_has_language_value_step
    REVERSIBLE = True
    choices = ("no", "yes")

    def prompt(self):
        if not has_columns_left(self.state):
            rich_print("No columns available to have a speaker column.")
            return "no"
        else:
            return get_response_from_menu_prompt(
                prompt_text="Does your data have a column/value for the language?",
                choices=self.choices,
            )

    def validate(self, response):
        return response in self.choices

    def effect(self):
        self.AUTOMATIC = not has_columns_left(self.state)
        if self.state[StepNames.data_has_language_value_step] == "yes":
            self.tour.add_step(
                LanguageHeaderStep(
                    name=StepNames.language_header_step,
                    prompt_text="These are the remaining values from the first row in your data. Which column contains the [bold blue]language?",
                    header_name="language",
                    state_subset=self.state_subset,
                ),
                self,
            )

        else:
            self.tour.add_step(SelectLanguageStep(state_subset=self.state_subset), self)


class SelectLanguageStep(Step):
    DEFAULT_NAME = StepNames.select_language_step
    REVERSIBLE = True

    def prompt(self):
        from g2p import get_arpabet_langs

        from everyvoice.text.phonemizer import AVAILABLE_G2P_ENGINES

        g2p_langs_full = get_arpabet_langs()[1]
        rich_print(
            "Note: if your dataset has more than one language in it, you will have to provide a 'language' column to indicate the language of each sample, because the configuration wizard can't guess!"
        )
        # TODO: currently we only support the languages from g2p, but we should add more
        supported_langs = list(AVAILABLE_G2P_ENGINES)
        supported_langs_choices = ["[und]: my language isn't here"] + [
            f"[{k}]: {g2p_langs_full.get(k, 'Unknown')}" for k in supported_langs
        ]
        return get_response_from_menu_prompt(
            choices=supported_langs_choices,
            title="Which of the following supported languages is the language of your dataset?",
            search=True,
        )

    def validate(self, response):
        return isinstance(response, str)

    def effect(self):
        self.saved_state = {
            "filelist_headers": copy(self.state["filelist_headers"]),
            "filelist_data": deepcopy(self.state.get("filelist_data", None)),
            "model_target_training_text_representation": None,
        }
        # Rename unselected headers to unknown:
        self.state["filelist_headers"] = rename_unknown_headers(
            self.state["filelist_headers"]
        )
        # re-parse data:
        reload_filelist_data_as_dict(self.state)
        # Add speaker IDs if they are not specified in the filelist
        if self.state[StepNames.data_has_speaker_value_step] == "no":
            add_missing_speaker(self.state)
        # Apply the language code:
        isocode = get_iso_code(self.response)
        # Apply text conversions and get target training representation
        self.state["model_target_training_text_representation"] = (
            apply_automatic_text_conversions(
                self.state["filelist_data"],
                self.state[StepNames.filelist_text_representation_step],
                global_isocode=isocode,
            )
        )


def add_missing_speaker(state):
    """Set all speakers IDs to the default speaker ID."""
    for item in state["filelist_data"]:
        item["speaker"] = state[StepNames.add_speaker_step]


def reload_filelist_data_as_dict(state):
    """Given a tour or step's state, reload the filelist_data as a dict if
    that was not already done."""
    data = state.get("filelist_data", None)
    if data and isinstance(data[0], dict):
        # data is already a dict, no need to do anything
        return

    # reparse data
    filelist_path = state.get(StepNames.filelist_step)
    if not isinstance(filelist_path, Path):
        filelist_path = Path(filelist_path).expanduser()
    if state.get(StepNames.filelist_format_step, None) in [
        "psv",
        "tsv",
        "csv",
    ]:
        headers = state["filelist_headers"]
        state["filelist_data"] = []
        for row in state["filelist_data_list"][1:]:
            item = {headers[i]: row[i] for i in range(len(row))}
            state["filelist_data"].append(item)
    assert isinstance(state["filelist_data"][0], dict)


class TextProcessingStep(Step):
    DEFAULT_NAME = StepNames.text_processing_step
    REVERSIBLE = True
    process_lookup = {
        0: {"fn": lower, "desc": "Lowercase"},
        1: {"fn": nfc_normalize, "desc": "NFC Normalization"},
    }

    def prompt(self):
        return get_response_from_menu_prompt(
            prompt_text=f"Which of the following text transformations would like to apply to your dataset's {self.state[StepNames.filelist_text_representation_step]}? See https://withblue.ink/2019/03/11/why-you-need-to-normalize-unicode-strings.html for information about NFC normalization.",
            choices=([process["desc"] for process in self.process_lookup.values()]),
            multi=True,
            return_indices=True,
        )

    def validate(self, response):
        return True

    def effect(self):
        from everyvoice.config.text_config import TextConfig

        self.saved_state = {}
        # Get Text Index
        if self.state.get("filelist_data_list", None):
            self.saved_state["filelist_data_list"] = deepcopy(
                self.state["filelist_data_list"]
            )
            text_index = self.state["filelist_headers"].index(
                self.state[StepNames.filelist_text_representation_step]
            )
            # Process global cleaners
            global_cleaners = TextConfig().cleaners
            for cleaner in global_cleaners:
                for i in tqdm(
                    range(len(self.state["filelist_data_list"])),
                    desc="Applying global default text normalization to data",
                ):
                    self.state["filelist_data_list"][i][text_index] = cleaner(
                        self.state["filelist_data_list"][i][text_index]
                    )
            # Process any dataset-specified cleaners
            for process in self.response:
                process_fn = self.process_lookup[process]["fn"]
                for i in tqdm(
                    range(len(self.state["filelist_data_list"])),
                    desc=f"Applying {self.process_lookup[process]['desc']} to data",
                ):
                    self.state["filelist_data_list"][i][text_index] = process_fn(
                        self.state["filelist_data_list"][i][text_index]
                    )
        else:
            self.saved_state["filelist_data"] = deepcopy(self.state["filelist_data"])
            # Process global cleaners
            global_cleaners = TextConfig().cleaners
            for cleaner in global_cleaners:
                for item in tqdm(
                    self.state["filelist_data"],
                    desc=f"Applying global default text normalization to '{self.state[StepNames.filelist_text_representation_step]}' data",
                ):
                    item[self.state[StepNames.filelist_text_representation_step]] = (
                        cleaner(
                            item[
                                self.state[StepNames.filelist_text_representation_step]
                            ]
                        )
                    )
            # Process any dataset-specified cleaners
            for process in self.response:
                process_fn = self.process_lookup[process]["fn"]
                for item in tqdm(
                    self.state["filelist_data"],
                    desc=f"Applying {self.process_lookup[process]['desc']} to '{self.state[StepNames.filelist_text_representation_step]}' data",
                ):
                    item[self.state[StepNames.filelist_text_representation_step]] = (
                        process_fn(
                            item[
                                self.state[StepNames.filelist_text_representation_step]
                            ]
                        )
                    )


class SoxEffectsStep(Step):
    DEFAULT_NAME = StepNames.sox_effects_step
    REVERSIBLE = True

    def prompt(self):
        return get_response_from_menu_prompt(
            prompt_text="Which of the following audio preprocessing options would you like to apply?",
            choices=(
                "Normalization (-3.0dB)",
                "Remove Silence at start and end",
                "Remove Silence throughout",
            ),
            multi=True,
            return_indices=True,
        )

    def validate(self, response):
        return True

    def effect(self):
        audio_effects = {
            0: [["norm", "-3.0"]],
            1: [
                [
                    "silence",
                    "1",
                    "0.1",
                    "0.1%",
                ],
                ["reverse"],  # reverse the clip to trim silence from end
                ["silence", "1", "0.1", "0.1%"],
                [
                    "reverse"
                ],  # reverse the clip again to revert to the right direction :)
            ],
            2: [["silence", "1", "0.1", "1.0%", "-1", "0.4", "1%"]],
        }
        self.state["sox_effects"] = [["channels", "1"]]
        if self.response:
            for effect in self.response:
                self.state["sox_effects"] += audio_effects[effect]

    def undo(self):
        del self.state["sox_effects"]
        super().undo()


class SymbolSetStep(Step):
    DEFAULT_NAME = StepNames.symbol_set_step
    AUTOMATIC = True
    REVERSIBLE = True

    def prompt(self):
        # TODO: This is a bit of a weird step, since it doesn't really prompt anything,
        #       it just applies the effect of trying to find character graphemes/phones.
        #       I'd still like to keep it here, since we might add more to this step in
        #       the future, and I don't want to lump the grapheme clustering logic into
        #       the effect of another step.
        rich_print(
            f"We will now read your entire dataset and try to determine the characters and/or phones in your dataset according to Unicode Grapheme clustering rules. Please carefully check your {TEXT_CONFIG_FILENAME_PREFIX}.yaml file (which is created at the end of the wizard) and adjust the symbol set as appropriate. If your language uses standard punctuation symbols to represent sounds, it is extra important that you go remove any of these symbols from the punctuation categories."
        )
        return True

    def validate(self, response):
        return bool(response)

    def effect(self):
        from everyvoice.config.text_config import Punctuation

        character_graphemes = set()
        phone_graphemes = set()
        for item in tqdm(
            self.state["filelist_data"], desc="Finding all symbols in your dataset"
        ):
            if "characters" in item:
                character_graphemes.update(guess_graphemes_in_text(item["characters"]))
            if "phones" in item:
                phone_graphemes.update(guess_ipa_phones_in_text(item["phones"]))
        character_graphemes.discard(" ")  # we don't want the space as a grapheme
        if not phone_graphemes and not character_graphemes:
            return
        punctuation = Punctuation().all
        symbols = {}
        if character_graphemes:
            symbols["characters"] = [
                x for x in sorted(list(character_graphemes)) if x not in punctuation
            ]
        if phone_graphemes:
            symbols["phones"] = [
                x for x in sorted(list(phone_graphemes)) if x not in punctuation
            ]
        self.state[StepNames.symbol_set_step] = symbols


def get_dataset_steps(dataset_index=0):
    return [
        FilelistStep(state_subset=f"dataset_{dataset_index}"),
        [
            DatasetPermissionStep(state_subset=f"dataset_{dataset_index}"),
            FilelistFormatStep(state_subset=f"dataset_{dataset_index}"),
            FilelistTextRepresentationStep(state_subset=f"dataset_{dataset_index}"),
            TextProcessingStep(state_subset=f"dataset_{dataset_index}"),
            HasSpeakerStep(state_subset=f"dataset_{dataset_index}"),
            HasLanguageStep(state_subset=f"dataset_{dataset_index}"),
            WavsDirStep(state_subset=f"dataset_{dataset_index}"),
            ValidateWavsStep(state_subset=f"dataset_{dataset_index}"),
            SymbolSetStep(state_subset=f"dataset_{dataset_index}"),
            SoxEffectsStep(state_subset=f"dataset_{dataset_index}"),
            DatasetNameStep(state_subset=f"dataset_{dataset_index}"),
        ],
    ]


if __name__ == "__main__":
    tour = Tour(
        name="Dataset Tour",
        # steps = [TextProcessingStep(name='test')]
        steps=get_dataset_steps(),
    )
    tour.visualize()
    tour.run()
