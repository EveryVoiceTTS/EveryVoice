import json
from pathlib import Path

import questionary
from rich import print
from rich.panel import Panel
from rich.style import Style

from everyvoice.config.preprocessing_config import Dataset, PreprocessingConfig
from everyvoice.config.shared_types import BaseTrainingConfig, LoggerConfig
from everyvoice.config.text_config import Symbols, TextConfig
from everyvoice.model.aligner.config import AlignerConfig
from everyvoice.model.e2e.config import E2ETrainingConfig, EveryVoiceConfig
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.utils import generic_psv_dict_reader, write_filelist
from everyvoice.wizard import CUSTOM_QUESTIONARY_STYLE, Step, StepNames
from everyvoice.wizard.dataset import return_dataset_steps
from everyvoice.wizard.prompts import get_response_from_menu_prompt
from everyvoice.wizard.utils import sanitize_path, write_dict_to_config

TEXT_CONFIG_FILENAME_PREFIX = "everyvoice-shared-text"
ALIGNER_CONFIG_FILENAME_PREFIX = "everyvoice-aligner"
PREPROCESSING_CONFIG_FILENAME_PREFIX = "everyvoice-shared-data"
TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX = "everyvoice-text-to-spec"
SPEC_TO_WAV_CONFIG_FILENAME_PREFIX = "everyvoice-spec-to-wav"
TEXT_TO_WAV_CONFIG_FILENAME_PREFIX = "everyvoice-text-to-wav"


class NameStep(Step):
    DEFAULT_NAME = StepNames.name_step

    def prompt(self):
        return input(
            "What would you like to call this project? This name should reflect the model you intend to train, e.g. 'my-sinhala-project' or 'english-french-model' or something similarly descriptive of your project: "
        )

    def validate(self, response):
        if len(response) == 0:
            print("Sorry, your project needs a name.")
            return False
        sanitized_path = sanitize_path(response)
        if not sanitized_path == response:
            print(
                f"Sorry, the project name '{response}' is not valid, since it will be used to create a folder and special characters are not permitted for folder names. Please re-type something like '{sanitized_path}' instead."
            )
            return False
        return True

    def effect(self):
        print(
            f"Great! Launching New Dataset Wizard 🧙 for project named '{self.response}'."
        )


class OutputPathStep(Step):
    DEFAULT_NAME = StepNames.output_step

    def prompt(self):
        return questionary.path(
            "Where should the New Dataset Wizard save your files?",
            default=".",
            style=CUSTOM_QUESTIONARY_STYLE,
        ).unsafe_ask()

    def validate(self, response):
        path = Path(response)
        if path.is_file():
            print(f"Sorry, '{path}' is a file. Please select a directory.")
            return False
        output_path = path / self.state.get(StepNames.name_step.value)
        if output_path.exists():
            print(
                f"Sorry, '{output_path}' already exists. Please choose another output directory or start again and choose a different project name."
            )
            return False
        return True

    def effect(self):
        output_path = Path(self.response) / self.state.get(StepNames.name_step.value)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"New Dataset Wizard 🧙 will put your files here: '{output_path}'")


class ConfigFormatStep(Step):
    DEFAULT_NAME = StepNames.config_format_step

    def prompt(self):
        return get_response_from_menu_prompt(
            "Which format would you like to output the configuration to?",
            ["yaml", "json"],
        )

    def validate(self, response):
        return response in ["yaml", "json"]

    def effect(self):
        output_path = (
            Path(self.state[StepNames.output_step.value])
            / self.state[StepNames.name_step.value]
        ).expanduser()
        # create_config_files
        config_dir = output_path / "config"
        config_dir.absolute().mkdir(exist_ok=True, parents=True)
        (config_dir / "../preprocessed").absolute().mkdir(parents=True, exist_ok=True)
        # log dir
        log_dir = config_dir / ".." / "logs_and_checkpoints"
        log_dir.absolute().mkdir(parents=True, exist_ok=True)
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
            wavs_dir = Path(
                self.state[dataset][StepNames.wavs_dir_step.value]
            ).expanduser()
            if not wavs_dir.is_absolute():
                if not output_path.is_absolute():
                    for _ in config_dir.parts:
                        wavs_dir = Path("..") / wavs_dir
                else:
                    wavs_dir = Path.cwd() / wavs_dir
            new_filelist_path = (
                Path("..")
                / f"{self.state[dataset][StepNames.dataset_name_step.value]}-filelist.psv"
            ).expanduser()
            for entry_i in range(len(self.state[dataset]["filelist_data"])):
                # Remove .wav if it was added to the basename
                if self.state[dataset]["filelist_data"][entry_i]["basename"].endswith(
                    ".wav"
                ):
                    self.state[dataset]["filelist_data"][entry_i][
                        "basename"
                    ] = self.state[dataset]["filelist_data"][entry_i][
                        "basename"
                    ].replace(
                        ".wav", ""
                    )
                self.state[dataset]["filelist_data"][entry_i] = {
                    k: v
                    for k, v in self.state[dataset]["filelist_data"][entry_i].items()
                    if k is not None and not k.startswith("unknown")
                }
            write_filelist(
                self.state[dataset]["filelist_data"],
                (config_dir / new_filelist_path).absolute(),
            )
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
        text_config_path = Path(f"{TEXT_CONFIG_FILENAME_PREFIX}.{self.response}")
        write_dict_to_config(
            json.loads(text_config.model_dump_json(exclude_none=True)),
            (config_dir / text_config_path).absolute(),
        )
        # Preprocessing Config
        preprocessed_training_filelist_path = (
            Path("..") / "preprocessed" / "training_filelist.psv"
        )
        preprocessed_validation_filelist_path = (
            Path("..") / "preprocessed" / "validation_filelist.psv"
        )
        preprocessing_config = PreprocessingConfig(
            dataset=self.state[StepNames.name_step.value],
            save_dir=Path("..") / "preprocessed",
            source_data=datasets,
        )
        preprocessing_config_path = Path(
            f"{PREPROCESSING_CONFIG_FILENAME_PREFIX}.{self.response}"
        )
        write_dict_to_config(
            json.loads(preprocessing_config.model_dump_json(exclude_none=True)),
            (config_dir / preprocessing_config_path).absolute(),
        )
        ## Create Aligner Config
        aligner_logger = LoggerConfig(
            name="AlignerExperiment", save_dir=log_dir
        ).model_dump()
        aligner_config = AlignerConfig(
            # This isn't the actual AlignerTrainingConfig, but we can use it because we just
            # inherit the defaults if we pass a dict to the AlignerConfig.training field
            training=BaseTrainingConfig(
                training_filelist=preprocessed_training_filelist_path,
                validation_filelist=preprocessed_validation_filelist_path,
                logger=aligner_logger,
            ).model_dump()
        )
        aligner_config_path = Path(f"{ALIGNER_CONFIG_FILENAME_PREFIX}.{self.response}")
        aligner_config_json = json.loads(
            aligner_config.model_dump_json(
                exclude_none=True, exclude={"preprocessing": True, "text": True}
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
        fp_logger = LoggerConfig(name="FeaturePredictionExperiment", save_dir=log_dir)
        fp_config = FeaturePredictionConfig(
            training=BaseTrainingConfig(
                training_filelist=preprocessed_training_filelist_path,
                validation_filelist=preprocessed_validation_filelist_path,
                logger=fp_logger,
            ).model_dump()
        )
        fp_config_path = Path(f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.{self.response}")
        fp_config_json = json.loads(
            fp_config.model_dump_json(
                exclude_none=True, exclude={"preprocessing": True, "text": True}
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
        vocoder_logger = LoggerConfig(name="VocoderExperiment", save_dir=log_dir)
        vocoder_config = VocoderConfig(
            training=BaseTrainingConfig(
                training_filelist=preprocessed_training_filelist_path,
                validation_filelist=preprocessed_validation_filelist_path,
                logger=vocoder_logger,
            ).model_dump()
        )
        vocoder_config_path = Path(
            f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.{self.response}"
        )
        vocoder_config_json = json.loads(
            vocoder_config.model_dump_json(
                exclude_none=True, exclude={"preprocessing": True}
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

        e2e_logger = LoggerConfig(name="E2E-Experiment", save_dir=log_dir)
        e2e_config = EveryVoiceConfig(
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
                exclude_none=True,
                exclude={"aligner": True, "feature_prediction": True, "vocoder": True},
            )
        )
        e2e_config_json["path_to_aligner_config_file"] = str(aligner_config_path)
        e2e_config_json["path_to_feature_prediction_config_file"] = str(fp_config_path)
        e2e_config_json["path_to_vocoder_config_file"] = str(vocoder_config_path)
        e2e_config_path = Path(f"{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}.{self.response}")
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
            ["no", "yes"],
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
