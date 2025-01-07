"""The main module for the wizard package."""

from enum import Enum

TEXT_CONFIG_FILENAME_PREFIX = "everyvoice-shared-text"
ALIGNER_CONFIG_FILENAME_PREFIX = "everyvoice-aligner"
PREPROCESSING_CONFIG_FILENAME_PREFIX = "everyvoice-shared-data"
TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX = "everyvoice-text-to-spec"
SPEC_TO_WAV_CONFIG_FILENAME_PREFIX = "everyvoice-spec-to-wav"
TEXT_TO_WAV_CONFIG_FILENAME_PREFIX = "everyvoice-text-to-wav"


class StepNames(Enum):
    name_step = "Name Step"
    contact_name_step = "Contact Name Step"
    contact_email_step = "Contact Email Step"
    dataset_name_step = "Dataset Name Step"
    dataset_permission_step = "Dataset Permission Step"
    output_step = "Output Path Step"
    wavs_dir_step = "Wavs Dir Step"
    filelist_step = "Filelist Step"
    filelist_format_step = "Filelist Format Step"
    validate_wavs_step = "Validate Wavs Step"
    filelist_text_representation_step = "Filelist Text Representation Step"
    target_training_representation_step = (
        "Target Training Representation Step"  # TODO: maybe don't need
    )
    data_has_header_line_step = "Filelist Has Header Line Step"
    basename_header_step = "Basename Header Step"
    text_header_step = "Text Header Step"
    data_has_speaker_value_step = "Data Has Speaker Step"
    speaker_header_step = "Speaker Header Step"
    know_speaker_step = "Know Speaker Step"
    add_speaker_step = "Add Speaker Step"
    data_has_language_value_step = "Data Has Language Step"
    language_header_step = "Language Header Step"
    select_language_step = "Select Language Step"
    text_processing_step = "Text Processing Step"
    g2p_step = "G2P Step"
    symbol_set_step = "Symbol-Set Step"
    sample_rate_config_step = "Sample Rate Config Step"
    audio_config_step = "Audio Config Step"
    sox_effects_step = "SoX Effects Step"
    more_datasets_step = "More Datasets Step"
    config_format_step = "Config Format Step"
