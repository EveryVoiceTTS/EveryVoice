import sys
from typing import List, Tuple, Union

import simple_term_menu
from rich import print
from rich.panel import Panel


def get_response_from_menu_prompt(
    prompt_text: str = "",
    choices: Tuple[str, ...] = (),
    title: str = "",
    multi=False,
    search=False,
    return_indices=False,
) -> Union[int, str, List[int], List[str]]:
    """Given some prompt text and a list of choices, create a simple terminal window
       and return the index of the choice

    Args:
        prompt_text (str): rich prompt text to print before menu
        choices (List[str]): choices to display

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
