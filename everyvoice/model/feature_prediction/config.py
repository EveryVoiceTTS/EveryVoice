from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.config import (
    FastSpeech2Config,
)

# If more models, change to Union[FastSpeech2Config, OtherModel]
FeaturePredictionConfig = FastSpeech2Config
