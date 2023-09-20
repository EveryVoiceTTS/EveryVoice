import json
from pathlib import Path

import questionary
from loguru import logger
from rich import print
from rich.panel import Panel
from rich.style import Style

from everyvoice.config.preprocessing_config import Dataset, PreprocessingConfig
from everyvoice.config.shared_types import BaseTrainingConfig, LoggerConfig
from everyvoice.config.text_config import Symbols, TextConfig
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.utils import generic_psv_dict_reader, write_filelist
from everyvoice.wizard import CUSTOM_QUESTIONARY_STYLE, Step, StepNames
from everyvoice.wizard.dataset import return_dataset_steps
from everyvoice.wizard.prompts import get_response_from_menu_prompt
from everyvoice.wizard.utils import sanitize_path, write_dict_to_config

TEXT_CONFIG_FILENAME_PREFIX = "everyvoice-text"
ALIGNER_CONFIG_FILENAME_PREFIX = "everyvoice-aligner"
PREPROCESSING_CONFIG_FILENAME_PREFIX = "everyvoice-preprocessing"
TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX = "everyvoice-text-to-spec"
SPEC_TO_WAV_CONFIG_FILENAME_PREFIX = "everyvoice-spec-to-wav"
TEXT_TO_WAV_CONFIG_FILENAME_PREFIX = "everyvoice-text-to-wav"


class NameStep(Step):
    def prompt(self):
        return input("What would you like to call this project? ")

    def validate(self, response):
        if len(response) == 0:
            logger.info("Sorry, you have to put something here")
            return False
        sanitized_path = sanitize_path(response)
        if not sanitized_path == response:
            logger.info(
                f"Sorry, your name: '{response}' is not valid, since it will be used to create a folder and special characters are not permitted for folder names. Please re-type something like '{sanitized_path}' instead."
            )
            return False
        return True

    def effect(self):
        logger.info(
            f"Great! Launching New Dataset Wizard 🧙 for project named '{self.response}'"
        )


class OutputPathStep(Step):
    def prompt(self):
        return questionary.path(
            "Where should the New Dataset Wizard save your files?",
            default=".",
            style=CUSTOM_QUESTIONARY_STYLE,
        ).ask()

    def validate(self, response):
        path = Path(response)
        if path.is_file():
            logger.warning(
                f"Sorry, the path at '{path.absolute()}' is a file. Please select a directory."
            )
            return False
        path = path / sanitize_path(self.state.get(StepNames.name_step.value))
        if path.exists():
            logger.warning(
                f"Sorry, the path at '{path.absolute()}' already exists. Please choose another output directory."
            )
            return False
        return True

    def effect(self):
        output_path = Path(self.response) / self.state.get(StepNames.name_step.value)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"New Dataset Wizard 🧙 will put your files here: '{output_path.absolute()}'"
        )


class ConfigFormatStep(Step):
    def prompt(self):
        return get_response_from_menu_prompt(
            "Which format would you like to output the configuration to?",
            ["yaml", "json"],
        )

    def validate(self, response):
        return response in ["yaml", "json"]

    def effect(self):
        output_path = (
            (
                Path(self.state[StepNames.output_step.value])
                / self.state[StepNames.name_step.value]
            )
            .expanduser()
            .absolute()
        )
        # create_config_files
        config_dir = (output_path / "config").absolute()
        config_dir.mkdir(exist_ok=True, parents=True)
        # log dir
        log_dir = LoggerConfig().save_dir.stem
        log_dir = (output_path / log_dir).absolute()
        log_dir.mkdir(parents=True, exist_ok=True)
        datasets = []
        # Text Configuration
        punctuation = []
        symbols = {}
        for dataset in [key for key in self.state.keys() if key.startswith("dataset_")]:
            # Gather Symbols for Text Configuration
            punctuation += self.state[dataset][
                StepNames.symbol_set_step.value
            ].punctuation
            symbols[f"{dataset}-symbols"] = self.state[dataset][
                StepNames.symbol_set_step.value
            ].symbol_set
            # Dataset Configs
            wavs_dir = (
                Path(self.state[dataset][StepNames.wavs_dir_step.value])
                .expanduser()
                .absolute()
            )
            new_filelist_path = (
                (
                    output_path
                    / f"{self.state[dataset][StepNames.dataset_name_step.value]}-filelist.psv"
                )
                .expanduser()
                .absolute()
            )
            for entry_i in range(len(self.state[dataset]["filelist_data"])):
                self.state[dataset]["filelist_data"][entry_i] = {
                    k: v
                    for k, v in self.state[dataset]["filelist_data"][entry_i].items()
                    if k is not None and not k.startswith("unknown")
                }
            write_filelist(self.state[dataset]["filelist_data"], new_filelist_path)
            sox_effects = self.state[dataset]["sox_effects"]
            filelist_loader = generic_psv_dict_reader

            datasets.append(
                Dataset(
                    label=dataset,
                    data_dir=wavs_dir,
                    filelist=new_filelist_path,
                    filelist_loader=filelist_loader,
                    sox_effects=sox_effects,
                )
            )
        text_config = TextConfig(
            symbols=Symbols(punctuation=list(set(punctuation)), **symbols)
        )
        text_config_path = (
            config_dir / f"{TEXT_CONFIG_FILENAME_PREFIX}.{self.response}"
        ).absolute()
        write_dict_to_config(
            json.loads(text_config.model_dump_json()), text_config_path
        )
        # Preprocessing Config
        preprocessed_training_filelist_path = (
            output_path / "preprocessed" / "training_filelist.psv"
        )
        preprocessed_validation_filelist_path = (
            output_path / "preprocessed" / "validation_filelist.psv"
        )
        preprocessing_config = PreprocessingConfig(
            dataset=self.state[StepNames.name_step.value],
            save_dir=(output_path / "preprocessed").absolute(),
            source_data=datasets,
        )
        preprocessing_config_path = (
            config_dir / f"{PREPROCESSING_CONFIG_FILENAME_PREFIX}.{self.response}"
        ).absolute()
        write_dict_to_config(
            json.loads(preprocessing_config.model_dump_json()),
            preprocessing_config_path,
        )
        ## Create Aligner Config
        aligner_logger = LoggerConfig(
            name="AlignerExperiment", save_dir=log_dir
        ).model_dump()
        aligner_config = AlignerConfig(
            # This isn't the actual AlignerTrainingConfig, but we can use it because we just
            # inherit the defaults if we pass a dict to the AlignerConfig.training field
            training=BaseTrainingConfig(
                training_filelist=preprocessed_training_filelist_path.absolute(),
                validation_filelist=preprocessed_validation_filelist_path.absolute(),
                logger=aligner_logger,
            ).model_dump()
        )
        aligner_config_path = (
            config_dir / f"{ALIGNER_CONFIG_FILENAME_PREFIX}.{self.response}"
        ).absolute()
        aligner_config_json = json.loads(aligner_config.model_dump_json())
        aligner_config_json["preprocessing"] = str(preprocessing_config_path)
        aligner_config_json["text"] = str(text_config_path)
        write_dict_to_config(aligner_config_json, aligner_config_path)
        # Create Feature Prediction Config
        fp_logger = LoggerConfig(name="FeaturePredictionExperiment", save_dir=log_dir)
        fp_config = FeaturePredictionConfig(
            training=BaseTrainingConfig(
                training_filelist=preprocessed_training_filelist_path.absolute(),
                validation_filelist=preprocessed_validation_filelist_path.absolute(),
                logger=fp_logger,
            ).model_dump()
        )
        fp_config_path = (
            config_dir / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.{self.response}"
        ).absolute()
        fp_config_json = json.loads(fp_config.model_dump_json())
        fp_config_json["preprocessing"] = str(preprocessing_config_path)
        fp_config_json["text"] = str(text_config_path)
        write_dict_to_config(fp_config_json, fp_config_path)
        # Create Vocoder Config
        vocoder_logger = LoggerConfig(name="VocoderExperiment", save_dir=log_dir)
        vocoder_config = VocoderConfig(
            training=BaseTrainingConfig(
                training_filelist=preprocessed_training_filelist_path.absolute(),
                validation_filelist=preprocessed_validation_filelist_path.absolute(),
                logger=vocoder_logger,
            ).model_dump()
        )
        vocoder_config_path = (
            config_dir / f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.{self.response}"
        ).absolute()
        vocoder_config_json = json.loads(vocoder_config.model_dump_json())
        vocoder_config_json["preprocessing"] = str(preprocessing_config_path)
        write_dict_to_config(vocoder_config_json, vocoder_config_path)
        # E2E Config
        e2e_logger = LoggerConfig(name="E2E-Experiment", save_dir=log_dir)
        e2e_config = EveryVoiceConfig(
            training=BaseTrainingConfig(
                training_filelist=preprocessed_training_filelist_path.absolute(),
                validation_filelist=preprocessed_validation_filelist_path.absolute(),
                logger=e2e_logger,
            ).model_dump(),
        )
        e2e_config_json = json.loads(e2e_config.model_dump_json())
        e2e_config_json["aligner"] = str(aligner_config_path)
        e2e_config_json["feature_prediction"] = str(fp_config_path)
        e2e_config_json["vocoder"] = str(vocoder_config_path)
        e2e_config_path = (
            config_dir / f"{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}.{self.response}"
        ).absolute()
        write_dict_to_config(e2e_config_json, e2e_config_path)
        print(
            Panel(
                f"You've finished configuring your dataset. Your files are located at {config_dir.absolute()}",
                title="Congratulations 🎉",
                subtitle="Next Steps Documentation: https://docs.everyvoice.ca/guides",
                border_style=Style(color="#0B4F19"),
            )
        )


class MoreDatasetsStep(Step):
    def prompt(self):
        return get_response_from_menu_prompt(
            "Do you have more datasets to process?",
            ["yes", "no"],
        )

    def validate(self, response):
        return response in ["yes", "no"]

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
            self.tour.add_step(
                MoreDatasetsStep(name=StepNames.more_datasets_step.value), self
            )
            for step in reversed(return_dataset_steps(dataset_index=new_dataset_index)):
                self.tour.add_step(step, self)
        else:
            self.tour.add_step(
                ConfigFormatStep(name=StepNames.config_format_step.value), self
            )
