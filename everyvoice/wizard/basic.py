import json
from copy import deepcopy
from pathlib import Path

import questionary
from email_validator import EmailNotValidError, validate_email
from rich import print as rich_print
from rich.panel import Panel
from rich.style import Style

from everyvoice.config.preprocessing_config import Dataset, PreprocessingConfig
from everyvoice.config.shared_types import (
    BaseTrainingConfig,
    ContactInformation,
    LoggerConfig,
    init_context,
)
from everyvoice.config.text_config import (
    DEFAULT_CLEANERS,
    LanguageBoundaries,
    Symbols,
    TextConfig,
)
from everyvoice.config.type_definitions import DatasetTextRepresentation
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.text.utils import is_sentence_final
from everyvoice.utils import generic_psv_filelist_reader, slugify, write_filelist
from everyvoice.wizard import (
    PREPROCESSING_CONFIG_FILENAME_PREFIX,
    SPEC_TO_WAV_CONFIG_FILENAME_PREFIX,
    TEXT_CONFIG_FILENAME_PREFIX,
    TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX,
    TEXT_TO_WAV_CONFIG_FILENAME_PREFIX,
    StepNames,
)
from everyvoice.wizard.dataset import TextProcessingStep, get_dataset_steps
from everyvoice.wizard.prompts import (
    CUSTOM_QUESTIONARY_STYLE,
    get_response_from_menu_prompt,
)
from everyvoice.wizard.tour import Step
from everyvoice.wizard.utils import (
    escape,
    ordered_intersection,
    sanitize_paths,
    write_dict_to_config,
)
from everyvoice.wizard.validators import validate_path


class NameStep(Step):
    DEFAULT_NAME = StepNames.name_step
    REVERSIBLE = True

    def prompt(self):
        rich_print(
            "What would you like to call this project? This name should reflect the model you intend to train, e.g. 'my-sinhala-project' or 'english-french-model' or something similarly descriptive of your project?"
        )
        return questionary.text(
            "project name: ", style=CUSTOM_QUESTIONARY_STYLE
        ).unsafe_ask()

    def validate(self, response):
        if len(response) == 0:
            rich_print("Sorry, your project needs a name. ")
            return False
        sanitized_path = slugify(response)
        if not sanitized_path == response:
            rich_print(
                f"Sorry, the project name '{escape(response)}' is not valid, since it will be used to create a folder and special characters are not permitted for folder names. Please re-type something like '{sanitized_path}' instead."
            )
            return False
        return True

    def effect(self):
        rich_print(
            f"Great! Launching Configuration Wizard 🧙 for project named '{self.response}'."
        )


class ContactNameStep(Step):
    DEFAULT_NAME = StepNames.contact_name_step
    REVERSIBLE = True

    def prompt(self):
        return questionary.text(
            "What is your full name? ", style=CUSTOM_QUESTIONARY_STYLE
        ).unsafe_ask()

    def validate(self, response):
        # Some languages don't use first and last names, so we can't necessarily check that response.split() > 1
        # It would be nice to have a better check here though.
        if len(response) < 3:
            rich_print("Sorry, EveryVoice requires a name to help prevent misuse.")
            return False
        return True

    def effect(self):
        rich_print(f"Great! Nice to meet you, '{escape(self.response)}'.")


class ContactEmailStep(Step):
    DEFAULT_NAME = StepNames.contact_email_step
    REVERSIBLE = True

    def prompt(self):
        return questionary.text(
            "Please provide a contact email address for your models. ",
            style=CUSTOM_QUESTIONARY_STYLE,
        ).unsafe_ask()

    def in_unit_testing(self):
        """Skip checking deliverability when in unit testing.

        Checking deliverability can be slow where there is not web connection, as
        is sometimes the case when running the unit tests, so skip it in that context.
        """
        import inspect

        return any(
            frame.filename.endswith("test_wizard.py") for frame in inspect.stack()
        )

    def validate(self, response):
        try:
            # Check that the email address is valid. Turn on check_deliverability
            # for first-time validations like on account creation pages (but not
            # login pages).
            validate_email(response, check_deliverability=not self.in_unit_testing())
        except EmailNotValidError as e:
            # The exception message is a human-readable explanation of why it's
            # not a valid (or deliverable) email address.
            rich_print("EveryVoice requires a valid email address to prevent misuse.")
            rich_print(str(e))
            return False
        return True

    def effect(self):
        emailinfo = validate_email(self.response, check_deliverability=False)
        email = emailinfo.normalized
        self.response = email
        rich_print(
            f"Great! Your contact email '{self.response}' will be saved to your models."
        )


class OutputPathStep(Step):
    DEFAULT_NAME = StepNames.output_step
    REVERSIBLE = True

    def prompt(self):
        return questionary.path(
            "Where should the Configuration Wizard save your files?",
            default=".",
            style=CUSTOM_QUESTIONARY_STYLE,
            only_directories=True,
        ).unsafe_ask()

    def sanitize_input(self, response):
        return sanitize_paths(response)

    def can_mkdir(self, path: Path) -> bool:
        """Make sure it's possible to create path, without leaving it behind."""
        dirs_to_make = []
        d = path
        while not d.exists() and d not in (Path("/"), Path(""), Path(".")):
            dirs_to_make.append(d)
            d = d.parent
        dirs_made = []
        try:
            for d in reversed(dirs_to_make):
                try:
                    d.mkdir()
                    dirs_made.append(d)
                except OSError as e:
                    rich_print(f"Sorry, could not create '{d}': {e}.")
                    return False
        finally:
            for d in reversed(dirs_made):
                d.rmdir()
        return True

    def validate(self, response) -> bool:
        path = Path(response)
        if path.is_file():
            rich_print(f"Sorry, '{path}' is a file. Please select a directory.")
            return False
        assert self.state is not None, "OutputPathStep requires NameStep"
        output_path = path / self.state.get(StepNames.name_step, "DEFAULT_NAME")
        if output_path.exists():
            rich_print(
                f"Sorry, '{output_path}' already exists. "
                "Please choose another output directory or start again and choose a different project name."
            )
            return False

        # We create the output directory in validate() instead of effect() so that
        # failure can be reported to the user and the question asked again if necessary.
        if not self.can_mkdir(output_path):
            rich_print("Please choose another output directory.")
            return False

        self.output_path = output_path
        return True

    def effect(self):
        rich_print(
            f"The Configuration Wizard 🧙 will put your files here: '{self.output_path}'"
        )


class OODDataStep(Step):
    """Ask user for the out-of-distribution (OOD) data source for a specific language."""

    REVERSIBLE = True
    choices = (
        "validation: use the validation split ... (warning: pollutes train/validation separation). ... If you don't intend to use StyleTTS2, or if you don't have any more data, please select this option.",
        "local: provide a path to a local plain-text file",
        "hf: download from a HuggingFace Hub repository",
    )

    def __init__(self, lang: str, **kwargs):
        super().__init__(**kwargs)
        self.lang = lang

    def prompt(self):
        rich_print(
            Panel(
                "OOD (out-of-distribution) texts are used by StyleTTS2 for calculating WavLM discriminator loss. "
                "They should come from outside your training and validation data and be the same language. "
                "It only needs to be text, you do not need accompanying audio for this part.",
                title=f"OOD Data for '{self.lang}'",
            )
        )
        return get_response_from_menu_prompt(
            f"Where should OOD reference texts for language '{self.lang}' come from?",
            self.choices,
        )

    def validate(self, response):
        return response in self.choices

    def effect(self):
        if self.response == self.choices[1]:  # local
            self.tour.add_step(
                OODLocalPathStep(
                    name=f"OOD Local Path Step [{self.lang}]",
                    lang=self.lang,
                ),
                self,
            )
        elif self.response == self.choices[2]:  # hf
            self.tour.add_step(
                OODHFRepoIDStep(
                    name=f"OOD HF Repo ID Step [{self.lang}]",
                    lang=self.lang,
                ),
                self,
            )
        else:  # validation
            self.saved_state = {
                "ood_raw_data": deepcopy(self.state.get("ood_raw_data"))
            }
            self.state["ood_raw_data"][self.lang] = {"source_type": "validation"}
            rich_print(
                Panel(
                    "[yellow]Warning: using the validation split as OOD data will pollute your "
                    "train/validation separation and may make your validation scores unreliable."
                    "Unless you are training your model for research purposes, this is probably OK."
                    "[/yellow]",
                )
            )


def _detect_ood_text_representation(first_line: str):
    """Return the DatasetTextRepresentation found in a PSV header line, or None.

    Prefers 'phones' over 'characters' when both are present.
    """
    fields = {f.strip() for f in first_line.split("|")}
    if DatasetTextRepresentation.ipa_phones.value in fields:
        return DatasetTextRepresentation.ipa_phones
    if DatasetTextRepresentation.characters.value in fields:
        return DatasetTextRepresentation.characters
    return None


class OODLocalPathStep(Step):
    """Collect a local file path for OOD data for a specific language."""

    REVERSIBLE = True

    def __init__(self, lang: str, **kwargs):
        super().__init__(**kwargs)
        self.lang = lang
        self._detected_repr = None

    def prompt(self):
        return questionary.path(
            f"Path to OOD plain-text file for language '{self.lang}': ",
            style=CUSTOM_QUESTIONARY_STYLE,
        ).unsafe_ask()

    def sanitize_input(self, response):
        return sanitize_paths(response)

    def validate(self, response) -> bool:
        if not validate_path(response, is_file=True, exists=True):
            return False
        try:
            with open(Path(response).expanduser(), encoding="utf-8") as fh:
                first_line = fh.readline()
        except OSError as e:
            rich_print(f"[red]Could not read the file: {e}[/red]")
            return False
        self._detected_repr = _detect_ood_text_representation(first_line)
        if self._detected_repr is None:
            rich_print(
                "[red]The file does not have a pipe-separated header with a "
                "'characters' or 'phones' column. Your text file should be a pipe separated file (basename|characters) and must have either a characters or phones column.[/red]"
            )
            return False
        return True

    def effect(self):
        self.saved_state = {"ood_raw_data": deepcopy(self.state.get("ood_raw_data"))}
        self.state["ood_raw_data"][self.lang] = {
            "source_type": "local",
            "local_path": self.response,
            "text_representation": (
                self._detected_repr.value
                if self._detected_repr is not None
                else DatasetTextRepresentation.characters.value
            ),
        }


class OODHFRepoIDStep(Step):
    """Collect a HuggingFace repository ID for OOD data for a specific language."""

    REVERSIBLE = True

    def __init__(self, lang: str, **kwargs):
        super().__init__(**kwargs)
        self.lang = lang

    def prompt(self):
        default = "everyvoice/StyleTTS2-English-OOD" if self.lang == "eng" else ""
        return questionary.text(
            f"HuggingFace repository ID for '{self.lang}' OOD data "
            "(e.g. 'everyvoice/StyleTTS2-English-OOD'): ",
            default=default,
            style=CUSTOM_QUESTIONARY_STYLE,
        ).unsafe_ask()

    def sanitize_input(self, response):
        return response.strip()

    def validate(self, response) -> bool:
        if not response or "/" not in response:
            rich_print(
                "Please enter a valid HuggingFace repository ID in the format 'owner/repo'."
            )
            return False

        from huggingface_hub import repo_exists

        if not repo_exists(response, repo_type="dataset"):
            rich_print(
                f"[red]Repository '[bold]{response}[/bold]' was not found on "
                "HuggingFace Hub. Please check the ID and try again.[/red]"
            )
            return False
        return True

    def effect(self):
        self.tour.add_step(
            OODHFFilenameStep(
                name=f"OOD HF Filename Step [{self.lang}]",
                lang=self.lang,
                repo_id=self.response,
            ),
            self,
        )


class OODHFFilenameStep(Step):
    """Collect the filename within a HuggingFace repository for OOD data."""

    REVERSIBLE = True

    def __init__(self, lang: str, repo_id: str, **kwargs):
        super().__init__(**kwargs)
        self.lang = lang
        self.repo_id = repo_id
        self._detected_repr = None

    def prompt(self):
        default = "OOD_texts.txt" if self.lang == "eng" else "ood.txt"
        return questionary.text(
            f"Filename within '{self.repo_id}' to use as OOD data: ",
            default=default,
            style=CUSTOM_QUESTIONARY_STYLE,
        ).unsafe_ask()

    def sanitize_input(self, response):
        return response.strip()

    def validate(self, response) -> bool:
        if not response:
            rich_print("Please enter a filename.")
            return False
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import (
                EntryNotFoundError,
                RepositoryNotFoundError,
            )

            local_path = Path(
                hf_hub_download(self.repo_id, repo_type="dataset", filename=response)
            )
            with open(local_path, encoding="utf-8") as fh:
                first_line = fh.readline()
            self._detected_repr = _detect_ood_text_representation(first_line)
            if self._detected_repr is None:
                rich_print(
                    f"[red]The file '[bold]{response}[/bold]' does not have a "
                    "pipe-separated header with a 'characters' or 'phones' column. "
                    "Please check the file format.[/red]"
                )
                return False
        except EntryNotFoundError:
            rich_print(
                f"[red]File '[bold]{response}[/bold]' was not found in "
                f"'[bold]{self.repo_id}[/bold]'. Please check the filename and try again.[/red]"
            )
            return False
        except RepositoryNotFoundError:
            rich_print(
                f"[yellow]Could not access repository '[bold]{self.repo_id}[/bold]'. "
                "Proceeding anyway.[/yellow]"
            )
            return False
        return True

    def effect(self):
        self.saved_state = {"ood_raw_data": deepcopy(self.state.get("ood_raw_data"))}
        self.state["ood_raw_data"][self.lang] = {
            "source_type": "hf",
            "repo_id": self.repo_id,
            "filename": self.response,
            "text_representation": (
                self._detected_repr.value
                if self._detected_repr is not None
                else DatasetTextRepresentation.characters.value
            ),
        }


class ConfigFormatStep(Step):
    DEFAULT_NAME = StepNames.config_format_step
    # ConfigFormatStep writes the results to disk and exits, so it's not reversible.
    REVERSIBLE = False

    def prompt(self):
        return get_response_from_menu_prompt(
            "Which format would you like to output the configuration to?",
            ("yaml", "json"),
        )

    def validate(self, response):
        return response in ("yaml", "json")

    def effect(self):  # noqa: C901
        from everyvoice.model.e2e.config import (
            E2EConfig,
            StyleTTS2ModelConfig,
            StyleTTS2TrainingConfig,
        )
        from everyvoice.model.feature_prediction.config import (
            FastSpeech2ModelConfig,
            FeaturePredictionConfig,
        )
        from everyvoice.utils import spinner

        assert self.state is not None
        with spinner("Preparing your configuration files"):
            output_path = (
                Path(self.state[StepNames.output_step])
                / self.state[StepNames.name_step]
            ).expanduser()
            # create_config_files
            config_dir = output_path / "config"
            config_dir.absolute().mkdir(exist_ok=True, parents=True)
            # preprocessed dir
            preprocessed_dir = output_path / "preprocessed"
            preprocessed_dir.absolute().mkdir(parents=True, exist_ok=True)
            # used in configs
            preprocessed_dir_relative_to_configs = Path("..") / "preprocessed"
            # log dir
            log_dir = output_path / "logs_and_checkpoints"
            log_dir.absolute().mkdir(parents=True, exist_ok=True)
            log_dir_relative_to_configs = Path("..") / "logs_and_checkpoints"
            datasets = []
            # Text Configuration
            symbols = {}
            multispeaker = False
            cache_speaker = None
            dataset_cleaners: dict[str, list] = {}  # map label->cleaners
            dataset_langs: dict[str, list[str]] = {}  # map label->lang codes
            for dataset in [
                key for key in self.state.keys() if key.startswith("dataset_")
            ]:
                dataset_state = self.state[dataset]
                # Get the name of the dataset, which is going to be its label
                dataset_name = dataset_state[StepNames.dataset_name_step]
                # Add Cleaners
                dataset_cleaners[dataset_name] = DEFAULT_CLEANERS + [
                    TextProcessingStep.process_lookup[x]["fn"]
                    for x in dataset_state.get(StepNames.text_processing_step, [])
                ]
                # Gather languages for per-language cleaner config and determining multilingual
                dataset_langs[dataset_name] = sorted(
                    set(item["language"] for item in dataset_state["filelist_data"])
                )
                # Gather Symbols for Text Configuration
                # rename keys based on dataset name:
                dataset_symbols = {
                    f"{dataset_name}_{k}": v
                    for k, v in dataset_state[StepNames.symbol_set_step].items()
                }
                symbols.update(dataset_symbols)
                # Check if the filelists has more than one distinct speaker and adjust Config corrspondingly
                if not multispeaker:
                    for item in dataset_state["filelist_data"]:
                        if (
                            item["speaker"] != cache_speaker
                            and cache_speaker is not None
                        ):
                            multispeaker = True
                            break
                        if cache_speaker is None:
                            cache_speaker = item["speaker"]
                # Dataset Configs
                wavs_dir = Path(dataset_state[StepNames.wavs_dir_step]).expanduser()
                if not wavs_dir.is_absolute():
                    if not output_path.is_absolute():
                        for _ in config_dir.parts:
                            wavs_dir = Path("..") / wavs_dir
                    else:
                        wavs_dir = Path.cwd() / wavs_dir
                new_filelist_path = (
                    Path("..") / f"{dataset_name}-filelist.psv"
                ).expanduser()
                filelist_data = dataset_state["filelist_data"]
                for i, entry in enumerate(filelist_data):
                    # Remove .wav if it was added to the basename
                    if entry["basename"].endswith(".wav"):
                        entry["basename"] = entry["basename"].replace(".wav", "")
                    # Remove unknown columns
                    filelist_data[i] = {
                        k: v
                        for k, v in entry.items()
                        if k is not None and not k.startswith("unknown")
                    }
                write_filelist(
                    filelist_data, (config_dir / new_filelist_path).absolute()
                )
                sox_effects = dataset_state["sox_effects"]
                filelist_loader = generic_psv_filelist_reader

                datasets.append(
                    Dataset(
                        label=dataset_name,
                        data_dir=wavs_dir,
                        filelist=new_filelist_path,
                        filelist_loader=filelist_loader,
                        sox_effects=sox_effects,
                        permissions_obtained=True,  # If you get this far, you've answered the Dataset Permission Attestation step correctly
                    )
                )

            if dataset_cleaners:
                global_cleaners = ordered_intersection(dataset_cleaners.values())
            else:
                global_cleaners = DEFAULT_CLEANERS

            # In the wizard, we initialize the language-specific cleaners for each
            # language in the data based as the intersection of the dataset cleaners
            # for datasets containing that language.
            language_codes = sorted(
                set(lang for langs in dataset_langs.values() for lang in langs)
            )
            multilingual = len(language_codes) > 1
            language_cleaners: dict[str, list] = {}
            for lang in language_codes:
                datasets_containing_lang = [
                    label for label, langs in dataset_langs.items() if lang in langs
                ]
                language_cleaners[lang] = ordered_intersection(
                    dataset_cleaners[label] for label in datasets_containing_lang
                )

            # Remove redundant cleaner definitions to make the output config leaner
            # and easier to read.
            for cleaners in language_cleaners.values():
                if cleaners != global_cleaners:
                    break  # found a non-redundant language cleaner -- keep them all
            else:
                language_cleaners.clear()  # all language cleaners are redundant

                # when language cleaners are all redundant, consider removing dataset cleaners too
                for cleaners in dataset_cleaners.values():
                    if cleaners != global_cleaners:
                        break  # found a non-redundant dataset cleaner -- keep them all
                else:
                    dataset_cleaners.clear()  # all dataset cleaners are redundant

            text_config = TextConfig(
                symbols=Symbols(**symbols),
                g2p_engines=self.state.get("custom_g2p", {}),
                cleaners=global_cleaners,
                language_cleaners=language_cleaners,
                dataset_cleaners=dataset_cleaners,
            )
            strong: str = "".join(
                [
                    char
                    for char in (
                        text_config.symbols.punctuation.question_symbols
                        + text_config.symbols.punctuation.periods
                        + text_config.symbols.punctuation.exclamations
                    )
                    if is_sentence_final(char)
                ]
            )
            weak: str = "".join(
                text_config.symbols.punctuation.commas
                + text_config.symbols.punctuation.semi_colons
                + text_config.symbols.punctuation.colons
            )
            text_config.boundaries = {
                lang: LanguageBoundaries(strong=strong, weak=weak)
                for lang in language_codes
            }
            text_config_path = Path(f"{TEXT_CONFIG_FILENAME_PREFIX}.{self.response}")
            write_dict_to_config(
                json.loads(text_config.model_dump_json(exclude_none=False)),
                (config_dir / text_config_path).absolute(),
            )
            # Contact
            CONTACT_INFO = ContactInformation(
                contact_name=self.state[StepNames.contact_name_step],
                contact_email=self.state[StepNames.contact_email_step],
            )

            with init_context({"writing_config": config_dir.resolve()}):
                # Preprocessing Config
                preprocessed_training_filelist_path = (
                    preprocessed_dir_relative_to_configs / "training_filelist.psv"
                )
                preprocessed_validation_filelist_path = (
                    preprocessed_dir_relative_to_configs / "validation_filelist.psv"
                )
                preprocessing_config = PreprocessingConfig(
                    dataset=self.state[StepNames.name_step],
                    save_dir=preprocessed_dir_relative_to_configs,
                    source_data=datasets,
                )
                preprocessing_config_path = Path(
                    f"{PREPROCESSING_CONFIG_FILENAME_PREFIX}.{self.response}"
                )
                write_dict_to_config(
                    json.loads(
                        preprocessing_config.model_dump_json(exclude_none=False)
                    ),
                    (config_dir / preprocessing_config_path).absolute(),
                )

                # Create Feature Prediction Config
                fp_logger = LoggerConfig(
                    name="FeaturePredictionExperiment",
                    save_dir=log_dir_relative_to_configs,
                )
                fp_config = FeaturePredictionConfig(
                    contact=CONTACT_INFO,
                    model=FastSpeech2ModelConfig(
                        multilingual=multilingual,
                        multispeaker=multispeaker,
                    ),
                    training=BaseTrainingConfig(
                        training_filelist=preprocessed_training_filelist_path,
                        validation_filelist=preprocessed_validation_filelist_path,
                        logger=fp_logger,
                    ).model_dump(),
                )
                fp_config_path = Path(
                    f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.{self.response}"
                )
                fp_config_json = json.loads(
                    fp_config.model_dump_json(
                        exclude_none=False,
                        exclude={"preprocessing": True, "text": True},
                    )
                )
                fp_config_json["path_to_preprocessing_config_file"] = str(
                    preprocessing_config_path
                )
                fp_config_json["path_to_text_config_file"] = str(text_config_path)
                write_dict_to_config(
                    fp_config_json,
                    (config_dir / fp_config_path).absolute(),
                )

                # Create Vocoder Config
                vocoder_logger = LoggerConfig(
                    name="VocoderExperiment", save_dir=log_dir_relative_to_configs
                )
                vocoder_config = VocoderConfig(
                    contact=CONTACT_INFO,
                    training=BaseTrainingConfig(
                        training_filelist=preprocessed_training_filelist_path,
                        validation_filelist=preprocessed_validation_filelist_path,
                        logger=vocoder_logger,
                    ).model_dump(),
                )
                vocoder_config_path = Path(
                    f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.{self.response}"
                )
                vocoder_config_json = json.loads(
                    vocoder_config.model_dump_json(
                        exclude_none=False, exclude={"preprocessing": True}
                    )
                )
                vocoder_config_json["path_to_preprocessing_config_file"] = str(
                    preprocessing_config_path
                )
                write_dict_to_config(
                    vocoder_config_json,
                    (config_dir / vocoder_config_path).absolute(),
                )

                from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.ev_config import (
                    OODDataHFSource,
                    OODDataSource,
                )

                e2e_logger = LoggerConfig(
                    name="E2E-Experiment", save_dir=log_dir_relative_to_configs
                )
                ood_raw_data_wizard = self.state.get("ood_raw_data", {})
                ood_raw_data_config = {}
                use_validation_as_ood = False
                for lang, data in ood_raw_data_wizard.items():
                    if data["source_type"] == "validation":
                        use_validation_as_ood = True
                    elif data["source_type"] == "hf":
                        ood_raw_data_config[lang] = OODDataSource(
                            hf=OODDataHFSource(
                                repo_id=data["repo_id"],
                                filename=data["filename"],
                            ),
                            text_representation=DatasetTextRepresentation(
                                data.get(
                                    "text_representation",
                                    DatasetTextRepresentation.characters.value,
                                )
                            ),
                        )
                    elif data["source_type"] == "local":
                        ood_raw_data_config[lang] = OODDataSource(
                            local_path=Path(data["local_path"]).expanduser(),
                            text_representation=DatasetTextRepresentation(
                                data.get(
                                    "text_representation",
                                    DatasetTextRepresentation.characters.value,
                                )
                            ),
                        )
                e2e_config = E2EConfig(
                    contact=CONTACT_INFO,
                    model=StyleTTS2ModelConfig(multispeaker=multispeaker),
                    preprocessing=preprocessing_config,
                    training=StyleTTS2TrainingConfig(
                        ood_raw_data=ood_raw_data_config,
                        use_validation_as_ood=use_validation_as_ood,
                        training_filelist=preprocessed_training_filelist_path,
                        validation_filelist=preprocessed_validation_filelist_path,
                        logger=e2e_logger,
                    ).model_dump(),
                )
                e2e_config_json = json.loads(
                    e2e_config.model_dump_json(
                        exclude_none=False,
                        exclude={"preprocessing": True, "text": True},
                    )
                )
                e2e_config_json["path_to_preprocessing_config_file"] = str(
                    preprocessing_config_path
                )
                e2e_config_json["path_to_text_config_file"] = str(text_config_path)
                e2e_config_path = Path(
                    f"{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}.{self.response}"
                )
                write_dict_to_config(
                    e2e_config_json,
                    (config_dir / e2e_config_path).absolute(),
                )

            rich_print(
                Panel(
                    f"You've finished configuring your dataset. Your files are located at {config_dir.absolute()}",
                    title="Congratulations 🎉",
                    subtitle="Next Steps Documentation: https://docs.everyvoice.ca/stable/guides",
                    border_style=Style(color="#0B4F19"),
                )
            )


class MoreDatasetsStep(Step):
    DEFAULT_NAME = StepNames.more_datasets_step
    REVERSIBLE = True

    def prompt(self):
        return get_response_from_menu_prompt(
            "Do you have more datasets to process?",
            ("no", "yes"),
        )

    def validate(self, response):
        return response in ("yes", "no")

    def effect(self):
        if self.response == "yes":
            new_dataset_index = (
                max(
                    [
                        int(key.split("_")[1])
                        for key in self.state.keys()
                        if key.startswith("dataset_")
                    ],
                    default=-1,
                )
                + 1
            )

            self.tour.add_steps(
                get_dataset_steps(dataset_index=new_dataset_index)
                + [MoreDatasetsStep()],
                self,
            )
        elif len([key for key in self.state.keys() if key.startswith("dataset_")]) == 0:
            rich_print("No dataset to save, exiting without saving any configuration.")
        else:
            language_codes = sorted(
                set(
                    item["language"]
                    for key in self.state
                    if key.startswith("dataset_")
                    for item in self.state[key].get("filelist_data", [])
                )
            )
            if "ood_raw_data" not in self.state:
                self.state["ood_raw_data"] = {}
            self.tour.add_steps(
                [
                    OODDataStep(
                        name=f"OOD Data Step [{lang}]",
                        lang=lang,
                    )
                    for lang in language_codes
                ]
                + [ConfigFormatStep(name=StepNames.config_format_step)],
                self,
            )

    def undo(self):
        if self.response == "yes":
            # delete the dataset from the state
            self.tour.remove_dataset(self.children[0].state_subset)  # type: ignore[misc]
        else:
            self.state.pop("ood_raw_data", None)
        super().undo()
