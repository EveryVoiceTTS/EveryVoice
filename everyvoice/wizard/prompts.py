"""
Encapsulate the logic for prompting the user for input in a simple terminal window
"""

import sys
from typing import Sequence

import rich
from questionary import Style
from rich.panel import Panel

try:
    import simple_term_menu
except NotImplementedError:  # pragma: no cover
    import everyvoice.wizard.simple_term_menu_win_stub as simple_term_menu

CUSTOM_QUESTIONARY_STYLE = Style(
    [
        ("qmark", "fg:default"),  # token in front of the question
        ("question", "bold"),  # question text
        ("answer", "fg:default"),  # submitted answer text behind the question
        ("pointer", "fg:default"),  # pointer used in select and checkbox prompts
        (
            "highlighted",
            "fg:default bold",
        ),  # pointed-at choice in select and checkbox prompts
        ("selected", "fg:default bold"),  # style for a selected item of a checkbox
        ("separator", "fg:default"),  # separator in lists
        ("instruction", "reverse"),  # user instructions for select, rawselect, checkbox
        ("text", "fg:default"),  # plain text
        ("disabled", "fg:default"),  # disabled choices for select and checkbox prompts
    ]
)


def get_response_from_menu_prompt(
    prompt_text: str = "",
    choices: Sequence[str] = (),
    title: str = "",
    multi=False,
    search=False,
    return_indices=False,
) -> str | int | list[str] | list[int]:
    """Given some prompt text and a list of choices, create a simple terminal window
       and return the index of the choice

    Args:
        prompt_text: rich prompt text to print in a Panel before the menu
        choices: choices to display
        title: plain text title to display before the menu (after prompt_text, if given)
        multi: if set, asks for multiple selections and returns an Interable of them
        search: if set, allow the user to search through the options with /
        return_indices: if set, return selected choice index(ices) instead of value(s)

    Returns:
        multi | return_indices | returns
        ----- | -------------- | -------
        false | false          | str: choice selected
        false | true           | int: index of choice selected
        true  | false          | list[str]: choices selected
        true  | true           | list[int]: indices of choices selected
    """
    if prompt_text:
        rich.print(Panel(prompt_text))
    # using TerminalMenu's title parameter truncates the title to the width of
    # the menu instead of wrapping it, so we use rich.print instead.
    if title:
        rich.print(title)
    menu = simple_term_menu.TerminalMenu(
        choices,
        multi_select=multi,
        multi_select_select_on_accept=(not multi),
        multi_select_empty_ok=multi,
        raise_error_on_interrupt=True,
        show_multi_select_hint=multi,
        show_search_hint=search,
        status_bar_style=("fg_gray", "bg_black"),
        quit_keys=(),
    )
    selection = menu.show()
    sys.stdout.write("\033[K")
    if multi:
        if selection is None:
            return []
        elif return_indices:
            return list(selection)  # selection might be a tuple, but we need a list
        else:
            return [choices[i] for i in selection]
    else:
        if return_indices:
            return selection
        else:
            return choices[selection]
