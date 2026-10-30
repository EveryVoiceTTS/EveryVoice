import typer
import typer.rich_utils
from rich.measure import Measurement


# See https://github.com/tiangolo/typer/issues/428#issuecomment-1238866548
class TyperGroupOrderAsDeclared(typer.core.TyperGroup):
    def list_commands(self, ctx):
        return self.commands.keys()


# Default arguments we always pass to the typer.Typer contructor.
# Usage: app = typer.Typer(**default_typer_args, help="...")
# To override an indivual value, use dictionary unpacking and merging:
# app = typer.Typer(**{**default_typer_args, "no_args_is_help": False}, help="...")
default_typer_args = {
    "pretty_exceptions_show_locals": False,
    "no_args_is_help": True,
    "context_settings": {"help_option_names": ["-h", "--help"]},
    "rich_markup_mode": "markdown",
    "cls": TyperGroupOrderAsDeclared,
}


def command(app: typer.Typer, no_args_is_help=True, **kwargs):
    """Wrapper around app.command setting no_args_is_help=True by default

    Usage: replace `@app.command(...)` by:

        @command(app, ...)

    When a command has no mandatory arguments or options, revert no_args_is_help:

        @command(app, no_args_is_help=False, ...)
    """
    return app.command(no_args_is_help=no_args_is_help, **kwargs)


# By default, typer does not enable breaking words to fold long lines in the
# help messages, so when we have URLs they just get truncated. Let's fix that.
# Hack written with the help of Claude Opus 5.5 medium, fully human reviewed and
# human tested.
# This abuses Typer internals, so let's guard it so EV doesn't break when Typer changes
class _FoldOverflow:
    """Render a Rich renderable with overflow="fold", so long words get broken.

    Needed for Markdown, which has no overflow attribute of its own but passes the
    render options down to the Text of each paragraph.
    """

    def __init__(self, renderable):
        self.renderable = renderable

    def __rich_console__(self, console, options):
        yield from console.render(self.renderable, options.update(overflow="fold"))

    def __rich_measure__(self, console, options):
        return Measurement.get(console, options, self.renderable)


if hasattr(typer.rich_utils, "_make_rich_text"):
    _orig_make_rich_text = typer.rich_utils._make_rich_text

    def _make_rich_text_fold(**kwargs):
        return _FoldOverflow(_orig_make_rich_text(**kwargs))

    typer.rich_utils._make_rich_text = _make_rich_text_fold
