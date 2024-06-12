import sys
from typing import Union

import simple_term_menu
from questionary import Style
from rich import print
from rich.panel import Panel

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
    choices: tuple[str, ...] = (),
    title: str = "",
    multi=False,
    search=False,
    return_indices=False,
) -> Union[int, str, list[int], list[str]]:
    """Given some prompt text and a list of choices, create a simple terminal window
       and return the index of the choice

    Args:
        prompt_text (str): rich prompt text to print before menu
        choices (list[str]): choices to display

    Returns:
        int: index of choice
    """
    if prompt_text:
        print(Panel(prompt_text))
    menu = simple_term_menu.TerminalMenu(
        choices,
        title=title,
        multi_select=multi,
        multi_select_select_on_accept=(not multi),
        multi_select_empty_ok=multi,
        raise_error_on_interrupt=True,
        show_multi_select_hint=multi,
        show_search_hint=search,
        status_bar_style=("fg_gray", "bg_black"),
    )
    index = menu.show()
    sys.stdout.write("\033[K")
    if index is None or return_indices:
        return index
    else:
        if isinstance(index, tuple):
            return [choices[i] for i in index]
        else:
            return choices[index]
