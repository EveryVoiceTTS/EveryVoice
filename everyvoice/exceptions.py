class InvalidConfiguration(Exception):
    def __init__(self, msg):
        super().__init__(self)
        self.msg = msg

    def __str__(self):
        return self.msg


class ConfigError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
