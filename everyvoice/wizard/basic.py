import json
from pathlib import Path

import questionary
from loguru import logger
from rich import print
from rich.panel import Panel
from rich.style import Style
from typing import List

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
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)
        self.config_dir: Path = None
        self.log_dir: Path = None
        self.aligner_config_filename: Path=None
        self.fp_config_filename: Path=None
        self.preprocessed_training_filelist_path: Path=None
        self.preprocessed_validation_filelist_path: Path=None
        self.preprocessing_config_filename: Path = None
        self.text_config_filename: Path=None
        self.vocoder_config_filename: Path=None

    def _aligner_config(self):
        """
        """
        ## Create Aligner Config
        aligner_logger = LoggerConfig(name="AlignerExperiment", save_dir=self.log_dir)
        aligner_config = AlignerConfig(
            training=BaseTrainingConfig(
                training_filelist=self.preprocessed_training_filelist_path,
                validation_filelist=self.preprocessed_validation_filelist_path,
                logger=aligner_logger,
            )
        )
        aligner_config_json = json.loads(aligner_config.json())
        aligner_config_json["preprocessing"] = str(self.preprocessing_config_filename)
        aligner_config_json["text"] = str(self.text_config_filename)
        self.aligner_config_filename = Path(f"aligner.{self.response}")
        write_dict_to_config(
                aligner_config_json,
                self.config_dir / self.aligner_config_filename)

    def _e2e_config(self):
        """
        """
        # E2E Config
        e2e_logger = LoggerConfig(name="E2E-Experiment", save_dir=self.log_dir)
        e2e_config = EveryVoiceConfig(
            training=BaseTrainingConfig(
                training_filelist=self.preprocessed_training_filelist_path,
                validation_filelist=self.preprocessed_validation_filelist_path,
                logger=e2e_logger,
            ),
        )
        e2e_config_json = json.loads(e2e_config.json())
        e2e_config_json["aligner"] = str(self.aligner_config_filename)
        e2e_config_json["feature_prediction"] = str(self.fp_config_filename)
        e2e_config_json["vocoder"] = str(self.vocoder_config_filename)
        e2e_config_path = (self.config_dir / f"e2e.{self.response}")
        write_dict_to_config(e2e_config_json, e2e_config_path)

    def _feature_prediction_config(self):
        """
        """
        # Create Feature Prediction Config
        fp_logger = LoggerConfig(name="FeaturePredictionExperiment", save_dir=self.log_dir)
        fp_config = FeaturePredictionConfig(
            training=BaseTrainingConfig(
                training_filelist=self.preprocessed_training_filelist_path,
                validation_filelist=self.preprocessed_validation_filelist_path,
                logger=fp_logger,
            )
        )
        fp_config_json = json.loads(fp_config.json())
        fp_config_json["preprocessing"] = str(self.preprocessing_config_filename)
        fp_config_json["text"] = str(self.text_config_filename)
        self.fp_config_filename = Path(f"feature_prediction.{self.response}")
        write_dict_to_config(
                fp_config_json,
                self.config_dir / self.fp_config_filename)

    def _preprocessing_config(self,
            output_path: Path,
            datasets: List,
            ):
        """
        """
        # Preprocessing Config
        self.preprocessed_training_filelist_path = (
            Path("..") / "preprocessed" / "training_filelist.psv"
        )
        self.preprocessed_validation_filelist_path = (
            Path("..") / "preprocessed" / "validation_filelist.psv"
        )
        #from pudb import set_trace; set_trace()
        preprocessing_config = PreprocessingConfig(
            dataset=self.state[StepNames.name_step.value],
            save_dir=str(Path("..") / "preprocessed"),
            source_data=datasets,
        )
        self.preprocessing_config_filename = Path(f"preprocessing.{self.response}")
        write_dict_to_config(
            json.loads(preprocessing_config.json()),
            self.config_dir / self.preprocessing_config_filename,
        )

    def _text_config(self,
            output_path: Path,
            ) -> List[Dataset]:
        """
        Write the TextConfig to a file.
        """
        # Text Configuration
        datasets = []
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
            )
            if not wavs_dir.is_absolute():
                wavs_dir = Path("../..") / wavs_dir
            new_filelist_path = (
                (
                    Path("..")
                    / f"{self.state[dataset][StepNames.dataset_name_step.value]}-filelist.psv"
                )
                .expanduser()
            )
            for entry_i in range(len(self.state[dataset]["filelist_data"])):
                self.state[dataset]["filelist_data"][entry_i] = {
                    k: v
                    for k, v in self.state[dataset]["filelist_data"][entry_i].items()
                    if k is not None and not k.startswith("unknown")
                }
            write_filelist(self.state[dataset]["filelist_data"],
                    self.config_dir / new_filelist_path)
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
        self.text_config_filename = Path(f"text.{self.response}")
        write_dict_to_config(
                json.loads(text_config.json()),
                self.config_dir / self.text_config_filename)

        return datasets

    def _vocoder_config(self):
        """
        """
        # Create Vocoder Config
        vocoder_logger = LoggerConfig(name="VocoderExperiment", save_dir=self.log_dir)
        vocoder_config = VocoderConfig(
            training=BaseTrainingConfig(
                training_filelist=self.preprocessed_training_filelist_path,
                validation_filelist=self.preprocessed_validation_filelist_path,
                logger=vocoder_logger,
            )
        )
        vocoder_config_json = json.loads(vocoder_config.json())
        vocoder_config_json["preprocessing"] = str(self.preprocessing_config_filename)
        self.vocoder_config_filename = Path(f"vocoder.{self.response}")
        write_dict_to_config(
                vocoder_config_json,
                self.config_dir / self.vocoder_config_filename)

    def prompt(self):
        return get_response_from_menu_prompt(
            "Which format would you like to output the configuration to?",
            ["yaml", "json"],
        )

    def validate(self, response):
        return response in ["yaml", "json"]

    def effect(self):
        # TODO Make paths arbitrary to the config file itself.
        # TODO testsuite that writes a config with relative path
        # TODO have a central place to make paths relative to their config.
        #      May be part of the model itself and not the wizard.
        #from pudb import set_trace; set_trace()
        output_path = (
            (
                Path(self.state[StepNames.output_step.value])
                / self.state[StepNames.name_step.value]
            )
            .expanduser()
        )
        # config directory
        self.config_dir = (output_path / "config")
        self.config_dir.absolute().mkdir(exist_ok=True, parents=True)
        # log directory
        self.log_dir = (output_path / "logs")
        self.log_dir.absolute().mkdir(parents=True, exist_ok=True)
        # preprocessed directory
        self.preprocessed_dir = (output_path / "preprocessed")
        self.preprocessed_dir.absolute().mkdir(parents=True, exist_ok=True)

        # create_config_files
        datasets = self._text_config(output_path=output_path)
        self._preprocessing_config(output_path=output_path, datasets=datasets)
        self._aligner_config()
        self._feature_prediction_config()
        self._vocoder_config()
        self._e2e_config()

        print(
            Panel(
                f"You've finished configuring your dataset. Your files are located at {self.config_dir.absolute()}",
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
