class EveryVoiceError(Exception):
    pass


class ConfigError(EveryVoiceError):
    pass


class OutOfVocabularySymbolError(EveryVoiceError):
    pass


class BadDataError(EveryVoiceError):
    pass


class ModelError(EveryVoiceError):
    pass


class DependencyError(EveryVoiceError):
    pass
