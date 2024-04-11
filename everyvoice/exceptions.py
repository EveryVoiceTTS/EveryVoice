class InvalidConfiguration(Exception):
    def __init__(self, msg):
        super().__init__(self)
        self.msg = msg

    def __str__(self):
        return self.msg


class ConfigError(Exception):
    pass


class OutOfVocabularySymbolError(Exception):
    pass


class BadDataError(Exception):
    pass
