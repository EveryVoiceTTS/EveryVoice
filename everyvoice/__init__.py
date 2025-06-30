import os
import sys

# Here, we add git submodule to path to allow imports to work as expected
# (e.g. from everyvoice.model.feature_prediction import FastSpeech2_lightning).
# This is similar to dynamically adding a __init__.py at the root of the submodule,
# which don't do for the following reasons:
#   - This would cause this error with mypy: https://github.com/python/mypy/issues/8944
#   - This would cause this error with setuptools and imports from within the submodule.
#     I.e. we want to be able to import like: from fs2.cli import app
# New submodules for other models should follow this pattern and be added here
(parent_folder_path, current_dir) = os.path.split(os.path.dirname(__file__))

sys.path.append(os.path.join(parent_folder_path, "model", "aligner", "wav2vec2aligner"))
sys.path.append(
    os.path.join(
        parent_folder_path, "model", "feature_prediction", "FastSpeech2_lightning"
    )
)
sys.path.append(
    os.path.join(parent_folder_path, "model", "vocoder", "HiFiGAN_iSTFT_lightning")
)
