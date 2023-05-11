import re
from functools import partial
from pathlib import Path
from unicodedata import normalize

import questionary
from loguru import logger
from slugify import slugify
from tqdm import tqdm

from everyvoice.config.text_config import Symbols
from everyvoice.utils import generic_csv_reader, generic_dict_loader, read_festival
from everyvoice.wizard import Step, StepNames, Tour
from everyvoice.wizard.prompts import get_response_from_menu_prompt
from everyvoice.wizard.validators import validate_path

# WAVS & FILELIST


class DatasetNameStep(Step):
    def prompt(self):
        return input("What would you like to call this dataset? ")

    def validate(self, response):
        if len(response) == 0:
            logger.info("Sorry, you have to put something here")
            return False
        slug = slugify(response)
        if not slug == response:
            logger.info(
                f"Sorry, your name: '{response}' is not valid, since it will be used to create a file and special characters are not permitted in filenames. Please re-type something like {slug} instead."
            )
            return False
        return True

    def effect(self):
        logger.info(
            f"Great! Configuration Wizard 🧙 finished the configuration for your dataset named '{self.response}'"
        )


class WavsDirStep(Step):
    def prompt(self):
        return questionary.path("Where are your audio files?").ask()

    def validate(self, response):
        valid_path = validate_path(response, is_dir=True, is_file=False, exists=True)
        if not valid_path:
            return False
        valid_path = Path(response).expanduser()
        contains_wavs = next(valid_path.glob("*.wav"), False)
        return valid_path and contains_wavs


class SampleRateConfigStep(Step):
    def prompt(self):
        return questionary.text(
            "What is the sample rate (in Hertz) of your data?"
        ).ask()

    def validate(self, response):
        try:
            self.response = int(response)
            return True
        except ValueError:
            logger.info(
                f"{response} is not a valid sample rate. Please enter an integer representing the sample rate in Hertz of your data."
            )
            return False


class FilelistFormatStep(Step):
    def prompt(self):
        return get_response_from_menu_prompt(
            "Select which format your filelist is in:",
            choices=["psv", "tsv", "csv", "festival"],
            multi=False,
            search=False,
            return_indices=False,
        )

    def validate(self, response):
        return isinstance(response, str)

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
    def __init__(self, name: str, prompt_text: str, header_name: str, **kwargs):
        super(HeaderStep, self).__init__(name=name, **kwargs)
        self.prompt_text = prompt_text
        self.header_name = header_name

    def prompt(self):
        choices = [
            f"{x}: {self.state['filelist_data'][0][x]}"
            for x in range(len(self.state["filelist_headers"]))
        ]
        # filter if already selected
        if "selected_headers" in self.state:
            choices = [
                x for x in choices if int(x[:1]) not in self.state["selected_headers"]
            ]
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


class FinalHeaderStep(HeaderStep):
    def effect(self):
        # Rename the filelist header with the standard header name
        index = self.state["filelist_headers"].index(self.response)
        self.state["filelist_headers"][index] = self.header_name
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


class HasSpeakerStep(Step):
    def prompt(self):
        if self.state[StepNames.filelist_format_step.value] == "festival":
            return "no"
        else:
            return get_response_from_menu_prompt(
                "Does your data have a column/value for the speaker?",
                ["yes", "no"],
            )

    def validate(self, response):
        return response in ["yes", "no"]

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
    def prompt(self):
        if self.state[StepNames.filelist_format_step.value] == "festival":
            return "no"
        else:
            return get_response_from_menu_prompt(
                "Does your data have a column/value for the language?",
                ["yes", "no"],
            )

    def validate(self, response):
        return response in ["yes", "no"]

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
    def prompt(self):
        from g2p import get_arpabet_langs

        logger.info(
            "Note: if your dataset has more than one language in it, you will have to add this information to your filelist, because the wizard can't guess!"
        )
        # TODO: currently we only support the languages from g2p, but we should add more
        supported_langs = get_arpabet_langs()[1]
        supported_langs_choices = ["[und]: my language isn't here"] + [
            f"[{k}]: {v}" for k, v in supported_langs.items()
        ]
        return get_response_from_menu_prompt(  # type: ignore
            "Which of the following supported languages are in your dataset?",
            supported_langs_choices,
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
    def prompt(self):
        return get_response_from_menu_prompt(
            "Which of the following text transformations would like to apply before determining the symbol set?",
            [
                "None",
                "Lowercase",
                "NFC Normalization - See here for more information: https://withblue.ink/2019/03/11/why-you-need-to-normalize-unicode-strings.html",
            ],
            multi=True,
            search=False,
            return_indices=True,
        )

    def validate(self, response):
        if isinstance(response, tuple) and (0 in response and len(response) > 1):
            logger.warning("Please either select None or one or more other steps.")
            return False
        else:
            return True

    def effect(self):
        process_lookup = {
            1: {"fn": lambda x: x.lower(), "desc": "lowercase"},
            2: {"fn": lambda x: normalize("NFC", x), "desc": ""},
        }
        for process in self.response:
            if process:
                for i in tqdm(
                    range(len(self.state["filelist_data"])),
                    desc=f"Applying {process_lookup[process]['desc']} to data",
                ):
                    self.state["filelist_data"][i]["text"] = process_lookup[process][
                        "fn"
                    ](self.state["filelist_data"][i]["text"])


class SoxEffectsStep(Step):
    def prompt(self):
        return get_response_from_menu_prompt(
            "Which of the following audio preprocessing options would you like to apply?",
            [
                "None",
                "Resample to suggested sample rate: 22050 kHz",
                "Normalization (-3.0dB)",
                "Remove Silence at Start",
                "Remove Silence throughout",
            ],
            multi=True,
            search=False,
            return_indices=True,
        )

    def validate(self, response):
        if isinstance(response, tuple) and (0 in response and len(response) > 1):
            logger.warning("Please either select None or one or more other steps.")
            return False
        else:
            return True

    def effect(self):
        audio_effects = {
            1: ["rate", "22050"],
            2: ["norm", "-3.0"],
            3: ["silence", "1", "0.1", "1.0%"],
            4: ["silence", "1", "0.1", "1.0%", "-1", "0.4", "1%"],
        }
        self.state["sox_effects"] = [["channel", "1"]]
        for effect in self.response:
            if effect:
                self.state["sox_effects"].append(audio_effects[effect])


class SymbolSetStep(Step):
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
            "Which of the following symbols are punctuation?",
            symbols,
            multi=True,
            search=True,
        )
        symbols = [x for x in symbols if x not in punctuation]
        banned_symbols = get_response_from_menu_prompt(  # type: ignore
            "Ignore utterances that contain any of the following characters",
            symbols,
            multi=True,
            search=True,
        )
        self.state["banned_symbols"] = banned_symbols
        symbols = [x for x in symbols if x not in banned_symbols]
        ignored_symbols = get_response_from_menu_prompt(  # type: ignore
            "Which of the following symbols can be ignored?",
            symbols,
            multi=True,
            search=True,
        )
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
        WavsDirStep(
            name=StepNames.wavs_dir_step.value, state_subset=f"dataset_{dataset_index}"
        ),
        Step(
            name=StepNames.filelist_step.value,
            prompt_method=questionary.path("Where is your data filelist?").ask,
            validate_method=partial(
                validate_path, is_dir=False, is_file=True, exists=True
            ),
            state_subset=f"dataset_{dataset_index}",
        ),
        FilelistFormatStep(
            name=StepNames.filelist_format_step.value,
            state_subset=f"dataset_{dataset_index}",
        ),
        HasSpeakerStep(
            name=StepNames.data_has_speaker_value_step.value,
            state_subset=f"dataset_{dataset_index}",
        ),
        HasLanguageStep(
            name=StepNames.data_has_language_value_step.value,
            state_subset=f"dataset_{dataset_index}",
        ),
        TextProcessingStep(
            name=StepNames.text_processing_step.value,
            state_subset=f"dataset_{dataset_index}",
        ),
        SymbolSetStep(
            name=StepNames.symbol_set_step.value,
            state_subset=f"dataset_{dataset_index}",
        ),
        SoxEffectsStep(
            name=StepNames.sox_effects_step.value,
            state_subset=f"dataset_{dataset_index}",
        ),
        DatasetNameStep(
            name=StepNames.dataset_name_step.value,
            state_subset=f"dataset_{dataset_index}",
        ),
    ]


if __name__ == "__main__":

    tour = Tour(
        name="Dataset Tour",
        # steps = [TextProcessingStep(name='test')]
        steps=return_dataset_steps(),
    )
    tour.visualize()
    tour.run()
