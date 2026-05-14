from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.ev_config import (  # noqa F401
    StyleTTS2Config,
    StyleTTS2ModelConfig,
    StyleTTS2PretrainedConfig,
    StyleTTS2TrainingConfig,
)

# If more models, change to Union[StyleTTS2Config, OtherModel]
E2EConfig = StyleTTS2Config
