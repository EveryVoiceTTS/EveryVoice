import os
import sys

# add git submodule to path to allow imports to work without adding a __init__.py at the root of the submodule, which causes this error with mypy: https://github.com/python/mypy/issues/8944
(parent_folder_path, current_dir) = os.path.split(os.path.dirname(__file__))

sys.path.append(
    os.path.join(parent_folder_path, "model", "aligner", "DeepForcedAligner")
)
sys.path.append(
    os.path.join(parent_folder_path, "model", "aligner", "FastSpeech2_lightning")
)
sys.path.append(
    os.path.join(parent_folder_path, "model", "aligner", "HiFiGAN_iSTFT_lightning")
)
