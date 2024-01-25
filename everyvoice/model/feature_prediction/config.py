from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.config import (  # noqa F401
    FastSpeech2Config,
    FastSpeech2ModelConfig,
)

# If more models, change to Union[FastSpeech2Config, OtherModel]
FeaturePredictionConfig = FastSpeech2Config
