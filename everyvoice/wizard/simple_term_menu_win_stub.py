"""
We use simple_term_menu to create multi-choice menus in the wizard, which we
like better than the questionary ones, but it does not support Windows.  This
file is just so I can run the tests and the wizard on Windows.
I know we don't support Windows because sox effects are not available on Windows,
but I (Eric Joanis) want to be able to develep the wizard and run unit tests on
my Windows machine, hence this file.

usage:

try:
    import simple_term_menu
except NotImplementedError:
    import simple_term_menu_win_stub as simple_term_menu

or

if os.name == "nt":
    import simple_term_menu_win_stub as simple_term_menu
else:
    import simple_term_menu
"""

import questionary


class TerminalMenu:
    """A stub class for the simple_term_menu.TerminalMenu class that supports the
    smallest feature subset required for the EveryVoice wizard to work on Windows.

    It's not nearly as pretty as what we do on Linux with the real thing, but
    it's good enough for development and testing."""

    def __init__(self, choices, multi_select, **_kwargs):
        self.choices = choices
        self.multi_select = multi_select

    def show(self) -> int | list[int]:
        """Show the menu and return the index/indices of the selected choice(s)."""
        if self.multi_select:
            responses = questionary.checkbox(
                message="",
                choices=self.choices,
            ).unsafe_ask()
            return [self.choices.index(response) for response in responses]
        else:
            response = questionary.select(
                message="",
                choices=self.choices,
            ).unsafe_ask()
            return self.choices.index(response)
