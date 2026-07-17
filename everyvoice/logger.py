"""loguru.logger with lazy importing

Usage: replace `from everyvoice import logger` by `from everyvoice import logger`
and then use logger.info(), logger.warning() and logger.error() as you would.
"""


def info(message, *args, **kwargs) -> None:
    from loguru import logger

    logger.opt(depth=1).info(message, *args, **kwargs)


def warning(message, *args, **kwargs) -> None:
    from loguru import logger

    logger.opt(depth=1).warning(message, *args, **kwargs)


def error(message, *args, **kwargs) -> None:
    from loguru import logger

    logger.opt(depth=1).error(message, *args, **kwargs)


def trace(message, *args, **kwargs) -> None:
    from loguru import logger

    logger.opt(depth=1).trace(message, *args, **kwargs)


def disable(name) -> None:
    from loguru import logger

    logger.disable(name)


def enable(name) -> None:
    from loguru import logger

    logger.enable(name)


def add(sink, **kwargs) -> int:
    from loguru import logger

    return logger.add(sink, **kwargs)


def remove(handler_id) -> None:
    from loguru import logger

    logger.remove(handler_id)
