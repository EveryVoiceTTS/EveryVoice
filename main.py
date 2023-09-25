""" This is a file for writing macros for mkdocs
    https://mkdocs-macros-plugin.readthedocs.io/en/latest/
"""

from everyvoice.wizard.basic import (
    ALIGNER_CONFIG_FILENAME_PREFIX,
    PREPROCESSING_CONFIG_FILENAME_PREFIX,
    SPEC_TO_WAV_CONFIG_FILENAME_PREFIX,
    TEXT_CONFIG_FILENAME_PREFIX,
    TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX,
    TEXT_TO_WAV_CONFIG_FILENAME_PREFIX,
)


def define_env(env):
    "Hook function"

    @env.macro
    def config_filename(filetype):
        if filetype == "aligner":
            return f"{ALIGNER_CONFIG_FILENAME_PREFIX}.yaml"
        if filetype == "preprocessing":
            return f"{PREPROCESSING_CONFIG_FILENAME_PREFIX}.yaml"
        if filetype == "spec-to-wav":
            return f"{SPEC_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml"
        if filetype == "text-to-wav":
            return f"{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml"
        if filetype == "text-to-spec":
            return f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
        if filetype == "text":
            return f"{TEXT_CONFIG_FILENAME_PREFIX}.yaml"
        raise ValueError(f"filetype: {filetype} does not exist.")
