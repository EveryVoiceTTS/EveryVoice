import json
from pathlib import Path

import questionary
from email_validator import EmailNotValidError, validate_email
from rich import print
from rich.panel import Panel
from rich.style import Style

from everyvoice.config.preprocessing_config import Dataset, PreprocessingConfig
from everyvoice.config.shared_types import (
    BaseTrainingConfig,
    ContactInformation,
    LoggerConfig,
    init_context,
)
from everyvoice.config.text_config import Symbols, TextConfig
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.e2e.config import E2ETrainingConfig, EveryVoiceConfig
from everyvoice.model.feature_prediction.config import (
    FastSpeech2ModelConfig,
    FeaturePredictionConfig,
)
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.utils import generic_psv_filelist_reader, slugify, write_filelist
from everyvoice.wizard import (
    ALIGNER_CONFIG_FILENAME_PREFIX,
    PREPROCESSING_CONFIG_FILENAME_PREFIX,
    SPEC_TO_WAV_CONFIG_FILENAME_PREFIX,
    TEXT_CONFIG_FILENAME_PREFIX,
    TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX,
    TEXT_TO_WAV_CONFIG_FILENAME_PREFIX,
    Step,
    StepNames,
)
from everyvoice.wizard.dataset import TextProcessingStep, get_dataset_steps
from everyvoice.wizard.prompts import (
    CUSTOM_QUESTIONARY_STYLE,
    get_response_from_menu_prompt,
)
from everyvoice.wizard.utils import write_dict_to_config


class NameStep(Step):
    DEFAULT_NAME = StepNames.name_step

    def prompt(self):
        return input(
            "What would you like to call this project? This name should reflect the model you intend to train, e.g. 'my-sinhala-project' or 'english-french-model' or something similarly descriptive of your project: "
        )

    def validate(self, response):
        if len(response) == 0:
            print("Sorry, your project needs a name. ")
            return False
        sanitized_path = slugify(response)
        if not sanitized_path == response:
            print(
                f"Sorry, the project name '{response}' is not valid, since it will be used to create a folder and special characters are not permitted for folder names. Please re-type something like '{sanitized_path}' instead."
            )
            return False
        return True

    def effect(self):
        print(
            f"Great! Launching Configuration Wizard 🧙 for project named '{self.response}'."
        )


class ContactNameStep(Step):
    DEFAULT_NAME = StepNames.contact_name_step

    def prompt(self):
        return input("What is your full name? ")

    def validate(self, response):
        # Some languages don't use first and last names, so we can't necessarily check that response.split() > 1
        # It would be nice to have a better check here though.
        if len(response) < 3:
            print("Sorry, EveryVoice requires a name to help prevent misuse.")
            return False
        return True

    def effect(self):
        print(f"Great! Nice to meet you, '{self.response}'.")


class ContactEmailStep(Step):
    DEFAULT_NAME = StepNames.contact_email_step

    def prompt(self):
        return input("Please provide a contact email address for your models. ")

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
            print("EveryVoice requires a valid email address to prevent misuse.")
            print(str(e))
            return False
        return True

    def effect(self):
        emailinfo = validate_email(self.response, check_deliverability=False)
        email = emailinfo.normalized
        self.response = email
        print(
            f"Great! Your contact email '{self.response}' will be saved to your models."
        )


class OutputPathStep(Step):
    DEFAULT_NAME = StepNames.output_step

    def prompt(self):
        return questionary.path(
            "Where should the Configuration Wizard save your files?",
            default=".",
            style=CUSTOM_QUESTIONARY_STYLE,
            only_directories=True,
        ).unsafe_ask()

    def sanitize_input(self, response):
        response = super().sanitize_input(response)
        return response.strip()

    def validate(self, response) -> bool:
        path = Path(response)
        if path.is_file():
            print(f"Sorry, '{path}' is a file. Please select a directory.")
            return False
        assert self.state is not None, "OutputPathStep requires NameStep"
        output_path = path / self.state.get(StepNames.name_step, "DEFAULT_NAME")
        if output_path.exists():
            print(
                f"Sorry, '{output_path}' already exists. "
                "Please choose another output directory or start again and choose a different project name."
            )
            return False

        # We create the output directory in validate() instead of effect() so that
        # failure can be reported to the user and the question asked again if necessary.
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            # we created it just to test permission, but don't leave it lying around in
            # case the wizard is interrupted or fails. We'll create it again when we save.
            output_path.rmdir()
        except OSError as e:
            print(
                f"Sorry, could not create '{output_path}': {e}. "
                "Please choose another output directory."
            )
            return False

        self.output_path = output_path
        return True

    def effect(self):
        print(
            f"The Configuration Wizard 🧙 will put your files here: '{self.output_path}'"
        )


class ConfigFormatStep(Step):
    DEFAULT_NAME = StepNames.config_format_step

    def prompt(self):
        return get_response_from_menu_prompt(
            "Which format would you like to output the configuration to?",
            ("yaml", "json"),
        )

    def validate(self, response):
        return response in ("yaml", "json")

    def effect(self):
        output_path = (
            Path(self.state[StepNames.output_step]) / self.state[StepNames.name_step]
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
        multilingual = False
        global_cleaners = (
            []
        )  # TODO: this should be fixed by https://github.com/roedoejet/EveryVoice/issues/359
        for dataset in [key for key in self.state.keys() if key.startswith("dataset_")]:
            dataset_state = self.state[dataset]
            # Add Cleaners
            # TODO: these should really be dataset-specific cleaners, not global cleaners
            # so this should be fixed by https://github.com/roedoejet/EveryVoice/issues/359
            if dataset_state.get(StepNames.text_processing_step):
                global_cleaners += [
                    TextProcessingStep().process_lookup[x]["fn"]
                    for x in dataset_state[StepNames.text_processing_step]
                ]
            # Gather Symbols for Text Configuration
            # rename keys based on dataset name:
            dataset_name = dataset_state[StepNames.dataset_name_step]
            dataset_symbols = {
                f"{dataset_name}_{k}": v
                for k, v in dataset_state[StepNames.symbol_set_step].items()
            }
            symbols.update(dataset_symbols)
            if dataset_state.get(StepNames.data_has_language_value_step, "no") == "yes":
                multilingual = True
            if dataset_state.get(StepNames.data_has_speaker_value_step, "no") == "yes":
                multispeaker = True
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
            write_filelist(filelist_data, (config_dir / new_filelist_path).absolute())
            sox_effects = dataset_state["sox_effects"]
            filelist_loader = generic_psv_filelist_reader

            datasets.append(
                Dataset(
                    label=dataset,
                    data_dir=wavs_dir,
                    filelist=new_filelist_path,
                    filelist_loader=filelist_loader,
                    sox_effects=sox_effects,
                    permissions_obtained=True,  # If you get this far, you've answered the Dataset Permission Attestation step correctly
                )
            )

        text_config = TextConfig(symbols=Symbols(**symbols))
        text_config.cleaners += global_cleaners
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
                json.loads(preprocessing_config.model_dump_json(exclude_none=False)),
                (config_dir / preprocessing_config_path).absolute(),
            )

            ## Create Aligner Config
            aligner_logger = LoggerConfig(
                name="AlignerExperiment", save_dir=log_dir_relative_to_configs
            ).model_dump()
            aligner_config = AlignerConfig(
                contact=CONTACT_INFO,
                # This isn't the actual AlignerTrainingConfig, but we can use it because we just
                # inherit the defaults if we pass a dict to the AlignerConfig.training field
                training=BaseTrainingConfig(
                    training_filelist=preprocessed_training_filelist_path,
                    validation_filelist=preprocessed_validation_filelist_path,
                    logger=aligner_logger,
                ).model_dump(),
            )
            aligner_config_path = Path(
                f"{ALIGNER_CONFIG_FILENAME_PREFIX}.{self.response}"
            )
            aligner_config_json = json.loads(
                aligner_config.model_dump_json(
                    exclude_none=False, exclude={"preprocessing": True, "text": True}
                )
            )
            aligner_config_json["path_to_preprocessing_config_file"] = str(
                preprocessing_config_path
            )
            aligner_config_json["path_to_text_config_file"] = str(text_config_path)
            write_dict_to_config(
                aligner_config_json,
                (config_dir / aligner_config_path).absolute(),
            )

            # Create Feature Prediction Config
            fp_logger = LoggerConfig(
                name="FeaturePredictionExperiment", save_dir=log_dir_relative_to_configs
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
                    exclude_none=False, exclude={"preprocessing": True, "text": True}
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

            # E2E Config
            e2e_logger = LoggerConfig(
                name="E2E-Experiment", save_dir=log_dir_relative_to_configs
            )
            e2e_config = EveryVoiceConfig(
                contact=CONTACT_INFO,
                aligner=aligner_config,
                feature_prediction=fp_config,
                vocoder=vocoder_config,
                training=E2ETrainingConfig(
                    training_filelist=preprocessed_training_filelist_path,
                    validation_filelist=preprocessed_validation_filelist_path,
                    logger=e2e_logger,
                ).model_dump(),
            )
            e2e_config_json = json.loads(
                e2e_config.model_dump_json(
                    exclude_none=False,
                    exclude={
                        "aligner": True,
                        "feature_prediction": True,
                        "vocoder": True,
                    },
                )
            )
            e2e_config_json["path_to_aligner_config_file"] = str(aligner_config_path)
            e2e_config_json["path_to_feature_prediction_config_file"] = str(
                fp_config_path
            )
            e2e_config_json["path_to_vocoder_config_file"] = str(vocoder_config_path)
            e2e_config_path = Path(
                f"{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}.{self.response}"
            )
            write_dict_to_config(
                e2e_config_json,
                (config_dir / e2e_config_path).absolute(),
            )

        print(
            Panel(
                f"You've finished configuring your dataset. Your files are located at {config_dir.absolute()}",
                title="Congratulations 🎉",
                subtitle="Next Steps Documentation: https://docs.everyvoice.ca/guides",
                border_style=Style(color="#0B4F19"),
            )
        )


class MoreDatasetsStep(Step):
    DEFAULT_NAME = StepNames.more_datasets_step

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
                    default=0,
                )
                + 1
            )

            self.tour.add_steps(
                get_dataset_steps(dataset_index=new_dataset_index)
                + [MoreDatasetsStep()],
                self,
            )
        elif len([key for key in self.state.keys() if key.startswith("dataset_")]) == 0:
            print("No dataset to save, exiting without saving any configuration.")
        else:
            self.tour.add_step(
                ConfigFormatStep(name=StepNames.config_format_step), self
            )
