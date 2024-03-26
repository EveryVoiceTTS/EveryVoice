import os
import re
from glob import glob
from pathlib import Path
from unicodedata import normalize

import questionary
from tqdm import tqdm

from everyvoice.config.preprocessing_config import DatasetTextRepresentation
from everyvoice.config.shared_types import TargetTrainingTextRepresentationLevel
from everyvoice.text.utils import (
    guess_graphemes_in_text_lines,
    guess_ipa_phones_in_text_lines,
)
from everyvoice.utils import generic_psv_filelist_reader, read_festival, slugify
from everyvoice.wizard import TEXT_CONFIG_FILENAME_PREFIX, Step, StepNames, Tour
from everyvoice.wizard.prompts import (
    CUSTOM_QUESTIONARY_STYLE,
    get_response_from_menu_prompt,
)
from everyvoice.wizard.utils import read_unknown_tabular_filelist
from everyvoice.wizard.validators import validate_path

# WAVS & FILELIST


class DatasetNameStep(Step):
    DEFAULT_NAME = StepNames.dataset_name_step

    def prompt(self):
        return input(
            "What would you like to call this dataset? This is needed because EveryVoice lets you train models with multiple sources of data. Please choose a name that distinguishes this data source, e.g. 'john-english' or 'maria-spanish' or something similarly descriptive: "
        )

    def validate(self, response):
        if len(response) == 0:
            print("Sorry, your dataset needs a name.")
            return False
        slug = slugify(response)
        if not slug == response:
            print(
                f"Sorry, your name: '{response}' is not valid, since it will be used to create a file and special characters are not permitted in filenames. Please re-type something like {slug} instead."
            )
            return False
        return True

    def effect(self):
        print(
            f"Great! The Configuration Wizard 🧙 finished the configuration for your dataset named '{self.response}'"
        )


class WavsDirStep(Step):
    DEFAULT_NAME = StepNames.wavs_dir_step

    def prompt(self):
        return questionary.path(
            "Where are your audio files?",
            style=CUSTOM_QUESTIONARY_STYLE,
            only_directories=True,
        ).unsafe_ask()

    def sanitize_input(self, response):
        response = super().sanitize_input(response)
        return response.strip()

    def validate(self, response) -> bool:
        valid_path = validate_path(response, is_dir=True, exists=True)
        if not valid_path:
            return False
        path_expanded = Path(response).expanduser()
        contains_wavs = (
            len(glob(os.path.join(path_expanded, "**/*.wav"), recursive=True)) > 0
        )
        if not contains_wavs:
            print(
                f"Sorry, no .wav files were found in '{path_expanded}'. Please choose a directory with audio files."
            )
        return valid_path and contains_wavs


class SampleRateConfigStep(Step):
    DEFAULT_NAME = StepNames.sample_rate_config_step

    def prompt(self):
        return questionary.text(
            "What is the sample rate (in Hertz) of your data?",
            style=CUSTOM_QUESTIONARY_STYLE,
        ).unsafe_ask()

    def validate(self, response):
        try:
            self.response = int(response)
            if self.response < 100 or float(response) != self.response:
                print(
                    f"{response} is not a valid sample rate. Please enter an integer representing the sample rate in Hertz of your data."
                )
                return False
            return True
        except ValueError:
            print(
                f"{response} is not a valid sample rate. Please enter an integer representing the sample rate in Hertz of your data."
            )
            return False


class FilelistStep(Step):
    DEFAULT_NAME = StepNames.filelist_step

    def prompt(self):
        return questionary.path(
            "Where is your data filelist?", style=CUSTOM_QUESTIONARY_STYLE
        ).unsafe_ask()

    def sanitize_input(self, response):
        response = super().sanitize_input(response)
        return response.strip()

    def validate(self, response) -> bool:
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
        assert self.state
        filelist_path = self.state.get(StepNames.filelist_step.value)
        initial_records = read_unknown_tabular_filelist(
            filelist_path, delimiter=separator, record_limit=10
        )

        column_count = len(initial_records[0])
        if column_count < 2:
            print(
                f"File '{filelist_path}' does not look like a '{file_type}' file: no record separator found on header line."
            )
            return False

        for i, record in enumerate(initial_records):
            if len(record) != column_count:
                print(
                    f"File '{filelist_path}' does not look like a '{file_type}' file: the {i}th record has a different number of fields than the header row."
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
                print(f"File '{filelist_path}' is not in the festival format.")
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
                        name=StepNames.text_header_step.value,
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
                        name=StepNames.basename_header_step.value,
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


class FilelistTextRepresentationStep(Step):
    DEFAULT_NAME = StepNames.filelist_text_representation_step
    text_representation_options = tuple(x.value for x in DatasetTextRepresentation)

    def prompt(self):
        return get_response_from_menu_prompt(
            prompt_text=f"Which representation is your text in? Choose '{DatasetTextRepresentation.ipa_phones.value}' if your text data only uses International Phonetic Alphabet characters (punctuation is also OK). Choose '{DatasetTextRepresentation.arpabet}' if your text data uses all ARPABET (punctuation is OK). Choose '{DatasetTextRepresentation.characters}' otherwise.",
            choices=self.text_representation_options,
            multi=False,
            search=False,
            return_indices=False,
        )

    def validate(self, response):
        return response in self.text_representation_options

    def effect(self):
        # Apply the text representation level as the new alias for text:
        for i, header in enumerate(self.state["filelist_headers"]):
            if header == "text":
                self.state["filelist_headers"][i] = self.response


# HEADER SELECTION


class HeaderStep(Step):
    DEFAULT_NAME = StepNames.text_header_step

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
            multi=False,
            search=False,
            return_indices=True,
        )
        return choice_indices[response]

    def validate(self, response):
        return isinstance(response, int)

    def effect(self):
        # Rename the filelist header with the standard header name
        if "selected_headers" not in self.state:
            self.state["selected_headers"] = []
        self.state["selected_headers"].append(self.response)
        self.state["filelist_headers"][self.response] = self.header_name


class HasHeaderLineStep(Step):
    DEFAULT_NAME = StepNames.data_has_header_line_step
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
        if self.state[StepNames.data_has_header_line_step.value] == "no":
            print("Reinterpreting your first row as a record, not headers.")
            self.state["filelist_data_list"].insert(
                0, self.state["filelist_data_list"][0]
            )


class HasSpeakerStep(Step):
    DEFAULT_NAME = StepNames.data_has_speaker_value_step
    choices = ("no", "yes")

    def prompt(self):
        if self.state[StepNames.filelist_format_step.value] == "festival":
            return "no"
        elif len(self.state.get("selected_headers", [])) >= len(
            self.state["filelist_data_list"][0]
        ):
            print("No columns left, we will assume you have no speaker column.")
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
    choices = ("no", "yes")

    def prompt(self):
        if self.state[StepNames.filelist_format_step.value] == "festival":
            return "no"
        elif len(self.state.get("selected_headers", [])) >= len(
            self.state["filelist_data_list"][0]
        ):
            print("No columns left, we will assume you have no language column.")
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

        from everyvoice.text.phonemizer import AVAILABLE_G2P_ENGINES

        g2p_langs_full = get_arpabet_langs()[1]
        print(
            "Note: if your dataset has more than one language in it, you will have to provide a 'language' column to indicate the language of each sample, because the configuration wizard can't guess!"
        )
        # TODO: currently we only support the languages from g2p, but we should add more
        supported_langs = list(AVAILABLE_G2P_ENGINES.keys())
        supported_langs_choices = ["[und]: my language isn't here"] + [
            f"[{k}]: {g2p_langs_full.get(k, 'Unknown')}" for k in supported_langs
        ]
        return get_response_from_menu_prompt(
            choices=supported_langs_choices,
            title="Which of the following supported languages is the language of your dataset?",
            multi=False,
            search=True,
        )

    def validate(self, response):
        return isinstance(response, str)

    def effect(self):
        from everyvoice.text.arpabet import ARPABET_TO_IPA_TRANSDUCER
        from everyvoice.text.phonemizer import AVAILABLE_G2P_ENGINES, get_g2p_engine

        # Rename unselected headers to unknown:
        for i, header in enumerate(self.state["filelist_headers"]):
            if header not in ["basename", "raw_text", "speaker", "language"] + [
                x.value for x in DatasetTextRepresentation
            ]:
                self.state["filelist_headers"][i] = f"unknown_{i}"
        # re-parse data:
        reload_filelist_data_as_dict(self.state)
        # Apply the language code:
        isocode = get_iso_code(self.response)
        g2p_engine = None
        if (
            isocode not in AVAILABLE_G2P_ENGINES
            and self.state[StepNames.filelist_text_representation_step.value]
            == DatasetTextRepresentation.characters.value
        ):
            print(
                f"Your text data contains characters and the language id {isocode} is not supported by a grapheme-to-phoneme engine. If you want to train using a pronunciation form (which usually results in better quality) you will have to add a g2p engine for your language. Please see <TODO:Docs> for more information."
            )
            self.state[
                "model_target_training_text_representation"
            ] = TargetTrainingTextRepresentationLevel.characters.value
        else:
            self.state[
                "model_target_training_text_representation"
            ] = TargetTrainingTextRepresentationLevel.ipa_phones.value
            g2p_engine = get_g2p_engine(isocode)
        for item in self.state["filelist_data"]:
            # add language code
            item["language"] = isocode
            # convert arpabet to phones
            if (
                DatasetTextRepresentation.arpabet.value in item
                and DatasetTextRepresentation.ipa_phones.value not in item
            ):
                item[
                    DatasetTextRepresentation.ipa_phones.value
                ] = ARPABET_TO_IPA_TRANSDUCER(
                    item[DatasetTextRepresentation.arpabet.value]
                ).output_string
            # if phones don't exist but g2p is available, calculate them
            if (
                DatasetTextRepresentation.characters.value in item
                and DatasetTextRepresentation.ipa_phones.value not in item
                and g2p_engine is not None
            ):
                phone_string_tokens = g2p_engine(
                    item[DatasetTextRepresentation.characters.value]
                )
                item[DatasetTextRepresentation.ipa_phones.value] = "".join(
                    phone_string_tokens
                )


def reload_filelist_data_as_dict(state):
    """Given a tour or step's state, reload the filelist_data as a dict if
    that was not already done."""
    data = state.get("filelist_data", None)
    if data and isinstance(data[0], dict):
        # data is already a dict, no need to do anything
        return

    # reparse data
    filelist_path = state.get(StepNames.filelist_step.value)
    if not isinstance(filelist_path, Path):
        filelist_path = Path(filelist_path).expanduser()
    if state.get(StepNames.filelist_format_step.value, None) in [
        "psv",
        "tsv",
        "csv",
    ]:
        state["filelist_data"] = generic_psv_filelist_reader(
            filelist_path,
            delimiter=state.get("filelist_delimiter"),
            fieldnames=state["filelist_headers"],
            file_has_header_line=(
                state.get(StepNames.data_has_header_line_step.value, "yes") == "yes"
            ),
        )
    else:
        state["filelist_data"] = read_festival(
            filelist_path,
            text_field_name=state.get(
                StepNames.filelist_text_representation_step.value, "text"
            ),
        )


def get_iso_code(language):
    if language is None:
        return None
    result = re.search(r"\[[\w-]*\]", language)
    if result is None:
        return language
    else:
        return result.group()[1:-1]


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
        # re-parse data if necessary:
        reload_filelist_data_as_dict(self.state)

        # Apply the selected text processing processes
        process_lookup = {
            0: {"fn": lambda x: x.lower(), "desc": "lowercase"},
            1: {"fn": lambda x: normalize("NFC", x), "desc": "NFC Normalization"},
        }
        if "symbols" not in self.state:
            self.state["symbols"] = {}
        if self.response:
            for process in self.response:
                process_fn = process_lookup[process]["fn"]
                for i in tqdm(
                    range(len(self.state["filelist_data"])),
                    desc=f"Applying {process_lookup[process]['desc']} to data",
                ):
                    self.state["filelist_data"][i][
                        self.state[StepNames.filelist_text_representation_step.value]
                    ] = process_fn(
                        self.state["filelist_data"][i][
                            self.state[
                                StepNames.filelist_text_representation_step.value
                            ]
                        ]
                    )


class SoxEffectsStep(Step):
    DEFAULT_NAME = StepNames.sox_effects_step

    def prompt(self):
        return get_response_from_menu_prompt(
            prompt_text="Which of the following audio preprocessing options would you like to apply?",
            choices=(
                "Resample to suggested sample rate: 22050 kHz",
                "Normalization (-3.0dB)",
                "Remove Silence at start and end",
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
            2: [
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
            3: ["silence", "1", "0.1", "1.0%", "-1", "0.4", "1%"],
        }
        self.state["sox_effects"] = [["channels", "1"]]
        if self.response:
            for effect in self.response:
                if (
                    effect == 2
                ):  # trimming leading and trailing silence is now a list of lists.
                    self.state["sox_effects"] += audio_effects[effect]
                else:
                    self.state["sox_effects"].append(audio_effects[effect])


class SymbolSetStep(Step):
    DEFAULT_NAME = StepNames.symbol_set_step

    def prompt(self):
        # TODO: This is a bit of a weird step, since it doesn't really prompt anything, it just applies the effect of trying to find
        #       character graphemes/phones. I'd still like to keep it here, since we might add more to this step in the future, and
        #       I don't want to lump the grapheme clustering logic into the effect of another step.
        print(
            f"We will now read your entire dataset and try to determine the characters and/or phones in your dataset according to Unicode Grapheme clustering rules. Please carefully check your {TEXT_CONFIG_FILENAME_PREFIX}.yaml file (which is created at the end of the wizard) and adjust the symbol set as appropriate. If your language uses standard punctuation symbols to represent sounds, it is extra important that you go remove any of these symbols from the punctuation categories."
        )
        return True

    def validate(self, response):
        return bool(response)

    def effect(self):
        # if characters guess graphemes
        character_graphemes = guess_graphemes_in_text_lines(
            [x.get("characters", "") for x in self.state["filelist_data"]]
        )
        phone_graphemes = guess_ipa_phones_in_text_lines(
            [x.get("phones", "") for x in self.state["filelist_data"]]
        )
        if not phone_graphemes and not character_graphemes:
            return
        symbols = {}
        if character_graphemes:
            symbols["characters"] = sorted(list(character_graphemes))
        if phone_graphemes:
            symbols["phones"] = sorted(list(phone_graphemes))
        self.state[StepNames.symbol_set_step.value] = symbols


def return_dataset_steps(dataset_index=0):
    return [
        WavsDirStep(state_subset=f"dataset_{dataset_index}"),
        FilelistStep(state_subset=f"dataset_{dataset_index}"),
        FilelistFormatStep(state_subset=f"dataset_{dataset_index}"),
        FilelistTextRepresentationStep(state_subset=f"dataset_{dataset_index}"),
        TextProcessingStep(state_subset=f"dataset_{dataset_index}"),
        HasSpeakerStep(state_subset=f"dataset_{dataset_index}"),
        HasLanguageStep(state_subset=f"dataset_{dataset_index}"),
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
