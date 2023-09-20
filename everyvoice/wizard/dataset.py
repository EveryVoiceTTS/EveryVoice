import os
import re
from glob import glob
from pathlib import Path
from unicodedata import normalize

import questionary
from loguru import logger
from tqdm import tqdm

from everyvoice.config.text_config import Symbols
from everyvoice.utils import generic_csv_reader, generic_dict_loader, read_festival
from everyvoice.wizard import CUSTOM_QUESTIONARY_STYLE, Step, StepNames, Tour
from everyvoice.wizard.prompts import get_response_from_menu_prompt
from everyvoice.wizard.validators import validate_path

# WAVS & FILELIST


class DatasetNameStep(Step):
    DEFAULT_NAME = StepNames.dataset_name_step

    def prompt(self):
        return input("What would you like to call this dataset? ")

    def validate(self, response):
        if len(response) == 0:
            logger.info("Sorry, you have to put something here")
            return False
        special_chars = re.compile(r"[\W]+")
        slug = re.sub(special_chars, "-", response)
        if not slug == response:
            logger.info(
                f"Sorry, your name: '{response}' is not valid, since it will be used to create a file and special characters are not permitted in filenames. Please re-type something like {slug} instead."
            )
            return False
        return True

    def effect(self):
        logger.info(
            f"Great! New Dataset Wizard 🧙 finished the configuration for your dataset named '{self.response}'"
        )


class WavsDirStep(Step):
    DEFAULT_NAME = StepNames.wavs_dir_step

    def prompt(self):
        return questionary.path(
            "Where are your audio files?", style=CUSTOM_QUESTIONARY_STYLE
        ).ask()

    def validate(self, response):
        valid_path = validate_path(response, is_dir=True, exists=True)
        if not valid_path:
            return False
        valid_path = Path(response).expanduser()
        contains_wavs = glob(os.path.join(valid_path, "**/*.wav"), recursive=True)
        return valid_path and contains_wavs


class SampleRateConfigStep(Step):
    DEFAULT_NAME = StepNames.sample_rate_config_step

    def prompt(self):
        return questionary.text(
            "What is the sample rate (in Hertz) of your data?",
            style=CUSTOM_QUESTIONARY_STYLE,
        ).ask()

    def validate(self, response):
        try:
            self.response = int(response)
            if self.response < 100 or float(response) != self.response:
                logger.info(
                    f"{response} is not a valid sample rate. Please enter an integer representing the sample rate in Hertz of your data."
                )
                return False
            return True
        except ValueError:
            logger.info(
                f"{response} is not a valid sample rate. Please enter an integer representing the sample rate in Hertz of your data."
            )
            return False


class FilelistStep(Step):
    DEFAULT_NAME = StepNames.filelist_step

    def prompt(self):
        return questionary.path(
            "Where is your data filelist?", style=CUSTOM_QUESTIONARY_STYLE
        ).ask()

    def validate(self, response):
        return validate_path(response, is_file=True, exists=True)


class FilelistFormatStep(Step):
    DEFAULT_NAME = StepNames.filelist_format_step
    separators = {"psv": "|", "tsv": "\t", "csv": ","}

    def prompt(self):
        return get_response_from_menu_prompt(
            prompt_text="Select which format your filelist is in:",
            choices=("psv", "tsv", "csv", "festival"),
            multi=False,
            search=False,
            return_indices=False,
        )

    def looks_like_sv(self, file_type, separator) -> bool:
        filelist_path = self.state.get(StepNames.filelist_step.value)
        initial_records = generic_csv_reader(
            filelist_path, delimiter=separator, record_limit=10
        )

        column_count = len(initial_records[0])
        if column_count < 2:
            logger.info(
                f"File {filelist_path} does not look like a {file_type} file: no record separator found on header line."
            )
            return False

        for i, record in enumerate(initial_records):
            if len(record) != column_count:
                logger.info(
                    f"File {filelist_path} does not look like a {file_type} file: the {i}th record has a different number of fields than the header row."
                )
                return False

        return True

    def validate(self, response):
        if response == "festival":
            filelist_path = self.state.get(StepNames.filelist_step.value)
            try:
                _ = read_festival(filelist_path, 10)
                return True
            except ValueError:
                logger.info(f"File {filelist_path} is not in the festival format.")
                return False

        separator = self.separators.get(response, None)
        if separator:
            return self.looks_like_sv(response, separator)

        assert False and "the above code covers all the accepted formats"

    def effect(self):
        """
        This effect occurs after the filelist format is selected. The Tour.state should now
        have both the filelist path and the filelist data type, so we parse the data and add
        the data to the tour state. We then inspect the headers for the data and add steps to
        select the basename and text if they are not already specified in the headers.
        """
        file_type = self.state.get(StepNames.filelist_format_step.value)
        filelist_path = self.state.get(StepNames.filelist_step.value)
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
            self.state["filelist_data"] = generic_csv_reader(
                filelist_path, delimiter=self.state.get("filelist_delimiter")
            )
            self.state["filelist_headers"] = list(self.state["filelist_data"][0])
            if "text" not in self.state["filelist_headers"]:
                self.tour.add_step(
                    HeaderStep(
                        name=StepNames.text_header_step.value,
                        prompt_text="Which column contains the [bold blue]text?",
                        header_name="text",
                        state_subset=self.state_subset,
                    ),
                    self,
                )
            if "basename" not in self.state["filelist_headers"]:
                self.tour.add_step(
                    HeaderStep(
                        name=StepNames.basename_header_step.value,
                        prompt_text="Which column contains the [bold blue]basenames?",
                        header_name="basename",
                        state_subset=self.state_subset,
                    ),
                    self,
                )


# HEADER SELECTION


class HeaderStep(Step):
    DEFAULT_NAME = StepNames.text_header_step

    def __init__(self, name: str, prompt_text: str, header_name: str, **kwargs):
        super(HeaderStep, self).__init__(name=name, **kwargs)
        self.prompt_text = prompt_text
        self.header_name = header_name

    def prompt(self):
        choices = [
            f"{x}: {self.state['filelist_data'][0][x]}"
            for x, _ in enumerate(self.state["filelist_headers"])
        ]
        # filter if already selected
        if "selected_headers" in self.state:
            choices = tuple(
                x for x in choices if int(x[:1]) not in self.state["selected_headers"]
            )
        response = get_response_from_menu_prompt(
            prompt_text=self.prompt_text,
            choices=choices,
            multi=False,
            search=False,
            return_indices=True,
        )
        # adjust index offset with previous selections
        if "selected_headers" in self.state:
            previous_selected = [
                x for x in self.state["selected_headers"] if x <= response
            ]
            response += len(previous_selected)
        return response

    def validate(self, response):
        return isinstance(response, int)

    def effect(self):
        # Rename the filelist header with the standard header name
        if "selected_headers" not in self.state:
            self.state["selected_headers"] = []
        self.state["selected_headers"].append(self.response)
        self.state["filelist_headers"][self.response] = self.header_name


class HasSpeakerStep(Step):
    DEFAULT_NAME = StepNames.data_has_speaker_value_step
    choices = ("yes", "no")

    def prompt(self):
        if self.state[StepNames.filelist_format_step.value] == "festival":
            return "no"
        else:
            return get_response_from_menu_prompt(
                prompt_text="Does your data have a column/value for the speaker?",
                choices=self.choices,
            )

    def validate(self, response):
        return response in self.choices

    def effect(self):
        if self.state[StepNames.data_has_speaker_value_step.value] == "yes":
            self.tour.add_step(
                HeaderStep(
                    name=StepNames.speaker_header_step.value,
                    prompt_text="These are the remaining values from the first row in your data. Which column contains the [bold blue]speaker?",
                    header_name="speaker",
                    state_subset=self.state_subset,
                ),
                self,
            )


class HasLanguageStep(Step):
    DEFAULT_NAME = StepNames.data_has_language_value_step
    choices = ("yes", "no")

    def prompt(self):
        if self.state[StepNames.filelist_format_step.value] == "festival":
            return "no"
        else:
            return get_response_from_menu_prompt(
                prompt_text="Does your data have a column/value for the language?",
                choices=self.choices,
            )

    def validate(self, response):
        return response in self.choices

    def effect(self):
        if self.state[StepNames.data_has_language_value_step.value] == "yes":
            self.tour.add_step(
                HeaderStep(
                    name=StepNames.language_header_step.value,
                    prompt_text="These are the remaining values from the first row in your data. Which column contains the [bold blue]language?",
                    header_name="language",
                    state_subset=self.state_subset,
                ),
                self,
            )
        else:
            self.tour.add_step(
                SelectLanguageStep(
                    name=StepNames.select_language_step.value,
                    state_subset=self.state_subset,
                ),
                self,
            )


class SelectLanguageStep(Step):
    DEFAULT_NAME = StepNames.select_language_step

    def prompt(self):
        from g2p import get_arpabet_langs

        logger.info(
            "Note: if your dataset has more than one language in it, you will have to add this information to your filelist, because the new dataset wizard can't guess!"
        )
        # TODO: currently we only support the languages from g2p, but we should add more
        supported_langs = get_arpabet_langs()[1]
        supported_langs_choices = ["[und]: my language isn't here"] + [
            f"[{k}]: {v}" for k, v in supported_langs.items()
        ]
        return get_response_from_menu_prompt(  # type: ignore
            choices=supported_langs_choices,
            title="Which of the following supported languages are in your dataset?",
            multi=False,
            search=True,
        )

    def validate(self, response):
        return isinstance(response, str)

    def effect(self):
        # Rename unselected headers to unknown:
        for i, header in enumerate(self.state["filelist_headers"]):
            if header not in ["basename", "text", "speaker", "language"]:
                self.state["filelist_headers"][i] = f"unknown_{i}"
        # re-parse data:
        filelist_path = self.state.get(StepNames.filelist_step.value)
        if not isinstance(filelist_path, Path):
            filelist_path = Path(filelist_path).expanduser()
        if self.state.get(StepNames.filelist_format_step.value, None) in [
            "psv",
            "tsv",
            "csv",
        ]:
            self.state["filelist_data"] = generic_dict_loader(
                filelist_path,
                delimiter=self.state.get("filelist_delimiter"),
                fieldnames=self.state["filelist_headers"],
            )
        else:
            self.state["filelist_data"] = read_festival(filelist_path)
        isocode = get_iso_code(self.response)
        for item in self.state["filelist_data"]:
            item["language"] = isocode


def get_iso_code(language):
    result = re.search(r"\[[\w-]*\]", language)
    if result is None:
        return language
    else:
        return result.group()[1:-1]


def return_symbols(language):
    # TODO: make actually return symbols from g2p
    if language is not None and language != "und":
        import string

        return set(string.ascii_letters)
    else:
        return None


class TextProcessingStep(Step):
    DEFAULT_NAME = StepNames.text_processing_step

    def prompt(self):
        return get_response_from_menu_prompt(
            prompt_text="Which of the following text transformations would like to apply before determining the symbol set?",
            choices=(
                "Lowercase",
                "NFC Normalization - See here for more information: https://withblue.ink/2019/03/11/why-you-need-to-normalize-unicode-strings.html",
            ),
            multi=True,
            search=False,
            return_indices=True,
        )

    def validate(self, response):
        return True

    def effect(self):
        process_lookup = {
            0: {"fn": lambda x: x.lower(), "desc": "lowercase"},
            1: {"fn": lambda x: normalize("NFC", x), "desc": ""},
        }
        if self.response is not None and len(self.response):
            for process in self.response:
                process_fn = process_lookup[process]["fn"]
                for i in tqdm(
                    range(len(self.state["filelist_data"])),
                    desc=f"Applying {process_lookup[process]['desc']} to data",
                ):
                    self.state["filelist_data"][i]["text"] = process_fn(
                        self.state["filelist_data"][i]["text"]
                    )


class SoxEffectsStep(Step):
    DEFAULT_NAME = StepNames.sox_effects_step

    def prompt(self):
        return get_response_from_menu_prompt(
            prompt_text="Which of the following audio preprocessing options would you like to apply?",
            choices=(
                "Resample to suggested sample rate: 22050 kHz",
                "Normalization (-3.0dB)",
                "Remove Silence at Start",
                "Remove Silence throughout",
            ),
            multi=True,
            search=False,
            return_indices=True,
        )

    def validate(self, response):
        return True

    def effect(self):
        audio_effects = {
            0: ["rate", "22050"],
            1: ["norm", "-3.0"],
            2: ["silence", "1", "0.1", "1.0%"],
            3: ["silence", "1", "0.1", "1.0%", "-1", "0.4", "1%"],
        }
        self.state["sox_effects"] = [["channel", "1"]]
        if self.response is not None and len(self.response):
            for effect in self.response:
                self.state["sox_effects"].append(audio_effects[effect])


class SymbolSetStep(Step):
    DEFAULT_NAME = StepNames.symbol_set_step

    def prompt(self):
        selected_language = get_iso_code(
            self.state.get(StepNames.select_language_step.value, None)
        )
        symbols_from_language = return_symbols(selected_language)
        all_tokens = None
        found_symbols = set(" ".join([x["text"] for x in self.state["filelist_data"]]))
        if all_tokens is None:
            logger.info(
                "We will now present all the symbols found in your data. You will have to answer which ones are punctuation, orthographic characters, and which ones can be ignored."
            )
            symbols = found_symbols
        else:
            symbols = found_symbols - symbols_from_language
            if symbols > 0:
                logger.info(
                    f"We found some characters that are not part of the standard symbol set for {selected_language}. You will have to answer which ones are punctuation, orthographic characters, and which ones can be ignored."
                )
        symbols = sorted(list(symbols))
        if not symbols:
            return
        punctuation = get_response_from_menu_prompt(  # type: ignore
            choices=symbols,
            title="Which of the following symbols are punctuation?",
            multi=True,
            search=True,
        )
        if punctuation is None:
            punctuation = []
        symbols = tuple(x for x in symbols if x not in punctuation)
        banned_symbols = get_response_from_menu_prompt(  # type: ignore
            title="Ignore utterances that contain any of the following characters:",
            choices=symbols,
            multi=True,
            search=True,
        )
        if banned_symbols is None:
            banned_symbols = []
        self.state["banned_symbols"] = banned_symbols
        symbols = tuple(x for x in symbols if x not in banned_symbols)
        ignored_symbols = get_response_from_menu_prompt(  # type: ignore
            title="Which of the following symbols can be ignored?",
            choices=symbols,
            multi=True,
            search=True,
        )
        if ignored_symbols is None:
            ignored_symbols = []
        return Symbols(
            punctuation=punctuation,
            symbol_set=[x for x in symbols if x not in ignored_symbols],
        )

    def validate(self, response):
        return isinstance(response, Symbols)

    def effect(self):
        if self.state["banned_symbols"]:
            banned_regexp = "|".join(self.state["banned_symbols"])
            removed = 0
            for i, item in tqdm(
                enumerate(self.state["filelist_data"]),
                desc="Removing items from your filelist with banned symbols",
            ):
                if re.match(banned_regexp, item["text"]):
                    del self.state["filelist_data"][i]
                    removed += 1
            logger.info(
                f"Removed {removed} samples from your data because they contained one of the following symbols: {self.state['banned_symbols']}"
            )


def return_dataset_steps(dataset_index=0):
    return [
        WavsDirStep(state_subset=f"dataset_{dataset_index}"),
        FilelistStep(state_subset=f"dataset_{dataset_index}"),
        FilelistFormatStep(state_subset=f"dataset_{dataset_index}"),
        HasSpeakerStep(state_subset=f"dataset_{dataset_index}"),
        HasLanguageStep(state_subset=f"dataset_{dataset_index}"),
        TextProcessingStep(state_subset=f"dataset_{dataset_index}"),
        SymbolSetStep(state_subset=f"dataset_{dataset_index}"),
        SoxEffectsStep(state_subset=f"dataset_{dataset_index}"),
        DatasetNameStep(state_subset=f"dataset_{dataset_index}"),
    ]


if __name__ == "__main__":
    tour = Tour(
        name="Dataset Tour",
        # steps = [TextProcessingStep(name='test')]
        steps=return_dataset_steps(),
    )
    tour.visualize()
    tour.run()
