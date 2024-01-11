import io
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Generator, Sequence, Union

from loguru import logger

from everyvoice.wizard import prompts

_NOTSET = object()  # Sentinel object for monkeypatch()


@contextmanager
def monkeypatch(obj, name, value) -> Generator:
    """Monkey patch obj.name to value for the duration of the context manager's life.

    Yields:
        value: the value monkey-patched, for use with "as v" notation"""
    saved_value = getattr(obj, name, _NOTSET)
    try:
        setattr(obj, name, value)
        yield value
    finally:
        if saved_value is _NOTSET:  # pragma: no cover
            delattr(obj, name)
        else:
            setattr(obj, name, saved_value)


QUIET = logging.CRITICAL + 100  # level high enough to disable all logging


@contextmanager
def patch_logger(
    module, level: int = logging.INFO
) -> Generator[logging.Logger, None, None]:
    """Monkey patch the logger for a given module with a unit testing logger
    of the given level. Use level=QUIET to just silence logging.

    Yields:
        logger (logging.Logger): patched logger, e.g., for use in self.assertLogs(logger)
    """
    with monkeypatch(module, "logger", logging.getLogger("UnitTesting")) as logger:
        logger.setLevel(level)
        yield logger


@contextmanager
def mute_logger(module: str) -> Generator[None, None, None]:
    """
    Temporarily mutes a module's `logger`.

    with mute_logger("everyvoice.base_cli.helpers"):
        config = FastSpeech2Config()
    """
    logger.disable(module)
    yield
    logger.enable(module)


@contextmanager
def capture_logs():
    """
    Context manager to capture log messages from loguru.
    """
    # [How to test loguru logger with unittest?](https://github.com/Delgan/loguru/issues/616)
    output = []
    handler_id = logger.add(output.append)
    yield output
    logger.remove(handler_id)


@contextmanager
def capture_stdout() -> Generator[io.StringIO, None, None]:
    """Context manager to capture what is printed to stdout.

    Usage:
        with capture_stdout() as stdout:
            # do stuff whose stdout you want to capture
        stdout.getvalue() is what was printed to stdout during the context

        with capture_stdout():
            # do stuff with stdout suppressed

    Yields:
        stdout (io.StringIO): captured stdout
    """
    f = io.StringIO()
    with redirect_stdout(f):
        yield f


@contextmanager
def capture_stderr():
    """Context manager to capture what is printed to stderr.

    Usage:
        with capture_stderr() as stderr:
            # do stuff whose stderr you want to capture
        stderr.getvalue() is what was printed to stderr during the context

        with capture_stderr():
            # do stuff with stderr suppressed

    Yields:
        stderr (io.StringIO): captured stderr
    """
    f = io.StringIO()
    with redirect_stderr(f):
        yield f


@contextmanager
def patch_menu_prompt(
    response_index: Union[int, list],
    multi=False,
) -> Generator[io.StringIO, None, None]:
    """Context manager to simulate what option(s) the user selects in a simple_term_menu.

    Args:
        response_index: the user's choice as a zero-based index for a
            single-option menu, or the list of indices for a multi-option
            menu
        multi: if True, response_index is a list of response_indices used one
            after the other each time a new simple_term_menu is instantiated

    Yields:
        stdout (io.StringIO): captured stdout stream.
    """
    with capture_stdout() as stdout:
        with monkeypatch(
            prompts, "simple_term_menu", SimpleTermMenuStub(response_index, multi)
        ):
            yield stdout


@contextmanager
def null_patch() -> Generator[None, None, None]:
    """dummy context manager when we must pass a monkeypatch but have nothing to patch"""
    yield


class Say:
    """Mock callable that returns response (if multi=False) or each value in
    response in turn (if multi=True) when it is called."""

    def __init__(self, response, multi=False) -> None:
        self.response = response
        self.last_index = -1
        self.multi = multi

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.multi:
            self.last_index += 1
            response = self.response[self.last_index]
        else:
            response = self.response
        if isinstance(response, BaseException):
            raise response
        return response


class SimpleTermMenuStub:
    """Stub class for the simple_term_menu module."""

    def __init__(self, response: Union[int, list[int]], multi=False):
        """Constructor

        Args:
            response: the index or indices the user will be simulated to choose
        """
        self.multi = multi
        self.last_index = -1
        self.response = response

    def TerminalMenu(self, *args, **kwargs):
        if not kwargs.get("raise_error_on_interrupt", False):  # pragma: no cover
            raise Exception(
                "raise_error_on_interrupt=True is required for TerminalMenu so we can receive and handle KeyboardInterrupt correctly"
            )
        return self

    def show(self):
        if self.multi:
            print("term stub", self.last_index, self.response[self.last_index + 1])
            self.last_index += 1
            response = self.response[self.last_index]
        else:
            response = self.response
        if isinstance(response, BaseException):
            raise response
        return response


class QuestionaryStub:
    """Stub class for the questionary module"""

    def __init__(self, responses: Sequence[str]) -> None:
        """Constructor

        Args:
            responses (str): the sequence of answers the user is simulated to provide
        """
        self.last_index = -1
        self.responses = responses

    def path(self, *args, **kwargs):
        return self

    text = path

    def ask(self):  # pragma: no cover
        # This will trigger a unit test failure if we use .ask()
        raise Exception(
            "Always use unsafe_ask() for questionary instances so that KeyboardInterrupt gets passed up to us."
        )

    def unsafe_ask(self):
        self.last_index += 1
        response = self.responses[self.last_index]
        if isinstance(response, BaseException):
            raise response
        return response
