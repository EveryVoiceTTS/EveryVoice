import json
from pathlib import Path

import questionary
from loguru import logger
from rich import print
from rich.panel import Panel
from rich.style import Style
from slugify import slugify

from everyvoice.config.preprocessing_config import Dataset, PreprocessingConfig
from everyvoice.config.shared_types import BaseTrainingConfig, LoggerConfig
from everyvoice.config.text_config import Symbols, TextConfig
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.e2e.config import EveryVoiceConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.utils import (
    generic_csv_loader,
    generic_psv_loader,
    generic_tsv_loader,
    read_festival,
)
from everyvoice.wizard import Step, StepNames
from everyvoice.wizard.dataset import return_dataset_steps
from everyvoice.wizard.prompts import get_response_from_menu_prompt
from everyvoice.wizard.utils import write_dict_to_config


class NameStep(Step):
    def prompt(self):
        return input("What would you like to call this project? ")

    def validate(self, response):
        return len(response) > 0

    def effect(self):
        logger.info(
            f"Great! Launching Configuration Wizard 🧙 for project named '{self.response}'"
        )


class OutputPathStep(Step):
    def prompt(self):
        return questionary.path(
            "Where should the wizard save your files?", default="."
        ).ask()

    def validate(self, response):
        path = Path(response)
        if path.is_file():
            logger.warning(
                f"Sorry, the path at '{path.absolute()}' is a file. Please select a directory."
            )
            return False
        path = path / slugify(self.state.get(StepNames.name_step.value))
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
            f"Configuration Wizard 🧙 will put your files here: '{output_path.absolute()}'"
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
        log_dir = (output_path / "logs").absolute()
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
            filelist_path = (
                Path(self.state[dataset][StepNames.filelist_step.value])
                .expanduser()
                .absolute()
            )
            sox_effects = self.state[dataset]["sox_effects"]
            if self.state[dataset][StepNames.filelist_format_step.value] == "psv":
                filelist_loader = generic_psv_loader
            elif self.state[dataset][StepNames.filelist_format_step.value] == "csv":
                filelist_loader = generic_csv_loader
            elif self.state[dataset][StepNames.filelist_format_step.value] == "tsv":
                filelist_loader = generic_tsv_loader
            elif (
                self.state[dataset][StepNames.filelist_format_step.value] == "festival"
            ):
                filelist_loader = read_festival
            datasets.append(
                Dataset(
                    label=dataset,
                    data_dir=wavs_dir,
                    filelist=filelist_path,
                    filelist_loader=filelist_loader,
                    sox_effects=sox_effects,
                )
            )
        text_config = TextConfig(
            symbols=Symbols(punctuation=list(set(punctuation)), **symbols)
        )
        text_config_path = (config_dir / f"text.{self.response}").absolute()
        write_dict_to_config(json.loads(text_config.json()), text_config_path)
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
            config_dir / f"preprocessing.{self.response}"
        ).absolute()
        write_dict_to_config(
            json.loads(preprocessing_config.json()), preprocessing_config_path
        )
        ## Create Aligner Config
        aligner_logger = LoggerConfig(name="AlignerExperiment", save_dir=log_dir)
        aligner_config = AlignerConfig(
            training=BaseTrainingConfig(
                training_filelist=preprocessed_training_filelist_path.absolute(),
                validation_filelist=preprocessed_validation_filelist_path.absolute(),
                logger=aligner_logger,
            )
        )
        aligner_config_path = (config_dir / f"aligner.{self.response}").absolute()
        aligner_config_json = json.loads(aligner_config.json())
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
            )
        )
        fp_config_path = (config_dir / f"feature_prediction.{self.response}").absolute()
        fp_config_json = json.loads(fp_config.json())
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
            )
        )
        vocoder_config_path = (config_dir / f"vocoder.{self.response}").absolute()
        vocoder_config_json = json.loads(vocoder_config.json())
        vocoder_config_json["preprocessing"] = str(preprocessing_config_path)
        write_dict_to_config(vocoder_config_json, vocoder_config_path)
        # E2E Config
        e2e_logger = LoggerConfig(name="E2E-Experiment", save_dir=log_dir)
        e2e_config = EveryVoiceConfig(
            training=BaseTrainingConfig(
                training_filelist=preprocessed_training_filelist_path.absolute(),
                validation_filelist=preprocessed_validation_filelist_path.absolute(),
                logger=e2e_logger,
            ),
        )
        e2e_config_json = json.loads(e2e_config.json())
        e2e_config_json["aligner"] = str(aligner_config_path)
        e2e_config_json["feature_prediction"] = str(fp_config_path)
        e2e_config_json["vocoder"] = str(vocoder_config_path)
        e2e_config_path = (config_dir / f"e2e.{self.response}").absolute()
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
                    ]
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
