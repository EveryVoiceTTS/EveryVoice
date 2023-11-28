Can we get shell completion in typer using pydantic's schema file?

# Shell Completion
https://opensource.com/article/18/3/creating-bash-completion-script


# CLI Libraries
[Other tools](https://github.com/mpkocher/pydantic-cli#other-related-tools)


# Typer
[Github](https://github.com/tiangolo/typer)
[Options Autocompletion](https://typer.tiangolo.com/tutorial/options-autocompletion/)
[typer.Argument with autocompletion](https://github.com/tiangolo/typer/issues/334)
https://github.com/tiangolo/typer/issues/334#issuecomment-1053548506
@patricksurry autocompletion works, but it is deprecated in click

DeprecationWarning: 'autocompletion' is renamed to 'shell_complete'. The old name is deprecated and will be removed in Click 8.1. See the docs about 'Parameter' for information about new behavior.
I'm still unable to make Arguments work with shell_complete.

On the opposite, shell_complete works very well with Options.

This is the situation, at least for me:

autocompleteshell_completion
Option✔️✔️
Argument✔️✖️


# PyDantic
https://docs.pydantic.dev/latest/api/pydantic_core/
[Use pydantic to define schema used across services?](https://github.com/pydantic/pydantic/issues/2261)


# Reddit
https://www.reddit.com/r/Python/comments/106lfm4/argdantic_typed_clis_with_argparse_and_pydantic/
typer: probably the most popular and main source of inspiration. It uses click instead of argparse, and does not yet support pydantic models to this day (it's a work in progress, afaik)


# ArgDantic:
https://github.com/edornd/argdantic


# CliDantic
https://github.com/edornd/clidantic
https://edornd.github.io/clidantic


# typer-CLI
https://typer.tiangolo.com/typer-cli/#typer-or-typer-cli


# Pydantic-CLI
https://github.com/mpkocher/pydantic-cli
https://github.com/mpkocher/pydantic-cli/blob/master/pydantic_cli/core.py


# Click
[Shell completion](https://click.palletsprojects.com/en/8.1.x/shell-completion/)
[Custom Type Completion](https://click.palletsprojects.com/en/8.1.x/shell-completion/#custom-type-completion)
