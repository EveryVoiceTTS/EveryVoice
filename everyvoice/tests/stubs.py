import builtins
import io
import logging
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Generator, Sequence, Union

from loguru import logger

from everyvoice.wizard import basic, dataset, prompts


class monkeypatch:
    """Monkey patch obj.name to value for the duration of the context manager's life.

    Yields:
        value: the value monkey-patched, for use with "as v" notation"""

    _NOTSET = object()  # Sentinel object for monkeypatch()

    def __init__(self, obj, name, value):
        self.obj = obj
        self.name = name
        self.value = value

    def __enter__(self):
        self.saved_value = getattr(self.obj, self.name, self._NOTSET)
        setattr(self.obj, self.name, self.value)
        return self.value

    def __exit__(self, *_exc_info):
        if self.saved_value is self._NOTSET:  # pragma: no cover
            delattr(self.obj, self.name)
        else:
            setattr(self.obj, self.name, self.saved_value)


class patch_logger:
    """Monkey patch the logger for a given module with a unit testing logger
    of the given level.

    Yields:
        logger (logging.Logger): patched logger, e.g., for use in self.assertLogs(logger)
    """

    def __init__(self, module, level: int = logging.INFO):
        self.monkey = monkeypatch(module, "logger", logging.getLogger("UnitTesting"))
        self.level = level

    def __enter__(self):
        logger = self.monkey.__enter__()
        logger.setLevel(self.level)
        return logger

    def __exit__(self, *_exc_info):
        self.monkey.__exit__(*_exc_info)


@contextmanager
def mute_logger(module: str) -> Generator[None, None, None]:
    """Temporarily mutes a module's `logger`.

    Usage:
        with mute_logger("everyvoice.base_cli.helpers"):
            config = FastSpeech2Config()
    """
    logger.disable(module)
    try:
        yield
    finally:
        logger.enable(module)


@contextmanager
def capture_logs():
    """
    Context manager to capture log messages from loguru.
    """
    # [How to test loguru logger with unittest?](https://github.com/Delgan/loguru/issues/616)
    output = []
    handler_id = logger.add(output.append)
    try:
        yield output
    finally:
        logger.remove(handler_id)


class capture_stdout:
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

    def __enter__(self):
        self.monkey = redirect_stdout(io.StringIO())
        return self.monkey.__enter__()

    def __exit__(self, *_exc_info):
        self.monkey.__exit__(*_exc_info)


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
def temp_chdir(path: Path) -> Generator[None, None, None]:
    """Context manager to temporarily change the current working directory.

    Args:
        path: the directory to change to
    """
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


class patch_menu_prompt:
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

    def __init__(self, response_index: Union[int, list], multi=False):
        self.response_index = response_index
        self.multi = multi

    def __enter__(self):
        self.monkey1 = monkeypatch(
            prompts,
            "simple_term_menu",
            SimpleTermMenuStub(self.response_index, self.multi),
        )
        self.monkey2 = capture_stdout()

        self.monkey1.__enter__()
        return self.monkey2.__enter__()

    def __exit__(self, *_exc_info):
        self.monkey2.__exit__(*_exc_info)
        self.monkey1.__exit__(*_exc_info)


class patch_input:
    """Shortcut for patching the builtin input() function, which we need often.

    Args: see class Say"""

    def __init__(self, response: Any, multi=False):
        self.response = response
        self.multi = multi

    def __enter__(self):
        self.monkey = monkeypatch(builtins, "input", Say(self.response, self.multi))
        return self.monkey.__enter__()

    def __exit__(self, *_exc_info):
        self.monkey.__exit__(*_exc_info)


class null_patch:
    """dummy context manager when we must pass a monkeypatch but have nothing to patch"""

    def __enter__(self):
        return None

    def __exit__(self, *_exc_info):
        pass


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

    def __init__(self, responses: Path | str | Sequence) -> None:
        """Constructor

        Args:
            responses: the (sequence of) answers the user is simulated to provide
        """
        self.last_index = -1
        self.responses: Sequence
        if isinstance(responses, (Path, str)):
            self.responses = [responses]
        else:
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
        if isinstance(response, Path):
            return str(response)
        return response


class patch_questionary:
    """Shortcut for monkey patching questionary everywhere

    Args: See QuestionaryStub"""

    def __init__(self, responses: Path | str | Sequence):
        self.responses = responses

    def __enter__(self):
        stub = QuestionaryStub(self.responses)
        module_name = "questionary"
        self.monkey1 = monkeypatch(basic, module_name, stub)
        self.monkey2 = monkeypatch(dataset, module_name, stub)
        self.monkey1.__enter__()
        return self.monkey2.__enter__()

    def __exit__(self, *_exc_info):
        self.monkey2.__exit__(*_exc_info)
        self.monkey1.__exit__(*_exc_info)
