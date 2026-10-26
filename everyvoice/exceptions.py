class EveryVoiceError(Exception):
    pass


class InvalidConfiguration(EveryVoiceError):
    pass


class ConfigError(EveryVoiceError):
    pass


class OutOfVocabularySymbolError(EveryVoiceError):
    pass


class BadDataError(EveryVoiceError):
    pass
