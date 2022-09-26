from config import CONFIGS
from preprocessor import Preprocessor

CONFIG = CONFIGS["lj"]

PREPROCESSOR = Preprocessor(CONFIG)


# Alignment preprocessing
# PREPROCESSOR.preprocess(
#     feature_prediction_filelist, process_sox_audio=True
# )
# Vocoder preprocessing

PREPROCESSOR.preprocess(
    output_path="preprocessed_filelist.psv",
    process_audio=True,
    process_spec=True,
    overwrite=True,
)
# # Feature prediction preprocessing
# PREPROCESSOR.preprocess(
#     feature_prediction_filelist,
#     process_spec=True,
#     process_energy=True,
#     process_f0=True,
#     process_duration=True,
#     process_pfs=True,
#     process_text=True,
# )
