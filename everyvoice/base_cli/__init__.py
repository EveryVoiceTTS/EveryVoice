import typer


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
