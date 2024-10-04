"""The main module for the wizard package."""

import sys
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

import questionary
import yaml
from anytree import PreOrderIter, RenderTree
from rich import print as rich_print
from rich.panel import Panel

from everyvoice._version import VERSION

from .prompts import CUSTOM_QUESTIONARY_STYLE, get_response_from_menu_prompt
from .utils import EnumDict as State
from .utils import NodeMixinWithNavigation, sanitize_paths

TEXT_CONFIG_FILENAME_PREFIX = "everyvoice-shared-text"
ALIGNER_CONFIG_FILENAME_PREFIX = "everyvoice-aligner"
PREPROCESSING_CONFIG_FILENAME_PREFIX = "everyvoice-shared-data"
TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX = "everyvoice-text-to-spec"
SPEC_TO_WAV_CONFIG_FILENAME_PREFIX = "everyvoice-spec-to-wav"
TEXT_TO_WAV_CONFIG_FILENAME_PREFIX = "everyvoice-text-to-wav"


class _Step:
    """_Step defines and documents the interface for step classes.

    Each step must implement the prompt and validate methods, and can optionally
    override the others.
    """

    def __init__(self, name: str):
        self.name = name

    def prompt(self):
        """Implement this method to prompt the user and return the step's response.

        This method must not have side effects, as it will not get called in resume mode.
        """
        raise NotImplementedError(
            f"This step ({self.name}) doesn't have a prompt method implemented. Please implement one."
        )

    def sanitize_input(self, response):
        """
        Override this method to perform data sanitization of user-provided input.
        """
        return response

    def validate(self, response) -> bool:
        """Each step class must implement this method to validate user responses."""
        raise NotImplementedError(
            f"This step ({self.name}) doesn't have a validate method implemented. Please implement one."
        )

    def effect(self):
        """Override this method to run additional code after the step resolves"""
        pass

    # declare `REVERSIBLE = True` in Step classes where the step's effects can be reversed.
    # If the step is reversible, implement undo() to reverse the effects of running the step.
    REVERSIBLE = False
    # declare `AUTOMATIC = True` in Step classes where the step does not prompt the user.
    AUTOMATIC = False

    def is_reversible(self):
        return self.REVERSIBLE

    def is_automatic(self):
        return self.AUTOMATIC

    def undo(self):
        """Implement undo() to reverse the effects of running the step.

        Note that the default implementation Step.undo() will work for most steps.
        When effect() modifies the state, add the previous value to self.saved_state
        When effect() adds children, they get removed automatically.
        So only override this method if you need to do more than that and
        is_reversible() returns True
        Don't forget to call super().undo() in subclasses.
        """
        pass


class Step(_Step, NodeMixinWithNavigation):
    """Step is a mixin to allow a tree-based interpretation of steps
    and it provides state management for the steps.

    Every step class must inherit from this class."""

    def __init__(
        self,
        name: None | Enum | str = None,
        default=None,
        parent=None,
        state_subset=None,
    ):
        if name is None:
            name = getattr(self, "DEFAULT_NAME", "default step name missing")
        name = name.value if isinstance(name, Enum) else name
        super().__init__(name)
        self.response = None
        self.completed = False
        self.default = default
        self.parent = parent
        self.state_subset = state_subset
        # state will be added when the Step is added to a Tour
        self.state: Optional[State] = None
        # tour will be added when the Step is added to a Tour
        self.tour: Optional[Tour] = None
        self._validation_failures = 0

    def __repr__(self) -> str:
        return f"{self.name}: {super().__repr__()}"

    def run(self, saved_response=None):
        """Prompt the user and save the response to the response attribute.
        If this method returns something truthy, continue, otherwise ask the prompt again.
        """
        if saved_response is not None:
            self.response = saved_response
        else:
            self.response = self.prompt()
        self.response = self.sanitize_input(self.response)
        if self.tour is not None and self.tour.trace:
            rich_print(f"{self.name}: '{self.response}'")
        if self.validate(self.response):
            self.completed = True
            try:
                if self.state is not None:
                    self.state[self.name] = self.response
                return self.response
            finally:
                self.effect()
        else:
            self._validation_failures += 1
            if self._validation_failures > 20:
                rich_print(
                    f"ERROR: {self.name} giving up after 20 validation failures."
                )
                sys.exit(1)
            self.run()

    def undo(self):
        """Basic undoing of the effects of running run() itself.

        Subclasses should further undo their own effects if any, then call super().undo().
        """
        self.response = None  # reset the response
        self.completed = False  # reset the completion flag
        if self.state:
            self.state.pop(self.name, None)  # remove the state entry for the step
        self.children = ()  # remove any children added by the step
        if saved_state := getattr(self, "saved_state", None):
            # restore the saved state if it exists
            for key, value in saved_state.items():
                if value is None:
                    self.state.pop(key, None)
                else:
                    self.state[key] = value
            del self.saved_state


class RootStep(Step):
    """Dummy step sitting at the root of the tour"""

    DEFAULT_NAME = "Root"
    REVERSIBLE = True

    def run(self, saved_response=None):
        pass

    def validate(self, response):
        return response is None


class Tour:
    def __init__(
        self,
        name: str,
        steps: list[Step],
        state: Optional[State] = None,
        trace: bool = False,
        debug_state: bool = False,
    ):
        """Create the tour by placing all steps under a dummy root node"""
        self.name = name
        self.state: State = state if state is not None else State()
        self.steps = steps
        self.trace = trace
        self.debug_state = debug_state
        self.root = RootStep()
        self.root.tour = self
        self.determine_state(self.root, self.state)
        self.add_steps(steps, self.root)

    def determine_state(self, step: Step, state: State):
        """Determines the state to use for the step based on the state subset"""
        if step.state_subset is not None:
            if step.state_subset not in state:
                state[step.state_subset] = State()
            step.state = state[step.state_subset]
        else:
            step.state = state

    def remove_dataset(self, state_subset: str):
        """Remove a dataset from the tour by state_subset"""
        if state_subset in self.state:
            del self.state[state_subset]

    def add_steps(self, steps: Sequence[Step | list[Step]], parent: Step):
        """Insert steps in front of the other children of parent.

        Steps are added as direct children.
        For sublists of steps, the first is a direct child, the rest are under it.
        """
        for item in reversed(steps):
            if isinstance(item, list):
                step, *children = item
                self.add_step(step, parent)
                self.add_steps(children, step)
            else:
                self.add_step(item, parent)

    def add_step(self, step: Step, parent: Step):
        """Insert a step in the specified position in the tour.

        Args:
            step: The step to add
            parent: The parent to add the step to
        """
        self.determine_state(step, self.state)
        step.tour = self
        children = list(parent.children)
        children.insert(0, step)
        parent.children = children

    def keyboard_interrupt_action(self, node):
        """Handle a keyboard interrupt by asking the user what to do"""
        action = None
        # Three Ctrl-C in a row, we just exist, but two in a row might be an
        # accident so ask again.
        for _ in range(2):
            try:
                action = get_response_from_menu_prompt(
                    "What would you like to do?",
                    [
                        "Go back one step",
                        "Continue",
                        "View progress",
                        "Save progress",
                        "Exit",
                    ],
                    return_indices=True,
                )
                break
            except KeyboardInterrupt:
                continue
        if action == 0:
            prev = node.prev()
            while prev.is_automatic():
                assert prev.is_reversible()
                prev.undo()
                prev = prev.prev()
                assert prev.is_reversible()
            if prev.is_reversible():
                prev.undo()
                return prev
            else:
                rich_print(
                    f"Sorry, the effects of the {prev.name} cannot be undone, continuing. If you need to go back, you'll have to restart the wizard."
                )
                return node
        elif action == 1:
            return node
        elif action == 2:
            self.visualize(highlight=node)
            return node
        elif action == 3:
            self.save_progress(node)
            return node
        else:  # still None, or the highest value in the choice list.
            sys.exit(1)

    def resume(self, resume_from: Path) -> Optional[Step]:
        """Resume the tour from a file containing the saved progress

        Returns: the node to continue from after applying the saved history.
        """

        try:
            with open(resume_from, "r", encoding="utf8") as f:
                q_and_a_list = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as e:
            rich_print(f"Error loading progress from {resume_from}: {e}")
            sys.exit(1)
        if (
            not isinstance(q_and_a_list, list)
            or not q_and_a_list
            or not all(
                isinstance(item, list) and len(item) == 2 for item in q_and_a_list
            )
        ):
            rich_print(
                f"Error loading progress from {resume_from}: invalid format. "
                "This does not look like a valid resume-from file. Aborting."
            )
            sys.exit(1)

        q_and_a_iter = iter(q_and_a_list)
        software, version = next(q_and_a_iter)
        if software != "EveryVoice Wizard" or version != VERSION:
            rich_print(
                f"[yellow]Warning: saved progress file is for {software} version '{version}', "
                f"but this is version '{VERSION}'. Proceeding anyway, but be aware that "
                "the saved responses may not be compatible.[/yellow]"
            )
        q_and_a = next(q_and_a_iter, None)
        node = self.root
        while node is not None and q_and_a is not None:
            saved_node_name, saved_response = q_and_a
            if saved_node_name.lower() != node.name.lower():
                rich_print(
                    f"Error: next tour question is {node.name} but resume list has {saved_node_name} instead.\n"
                    "Your resume-from file is likely out of sync. Aborting."
                )
                sys.exit(1)
            if node.validate(saved_response):
                if node.name != "Root":
                    rich_print(
                        f"Applying saved response '{saved_response}' for [green]{node.name}[/green]"
                    )
                node.run(saved_response=saved_response)
            else:
                rich_print(
                    Panel(
                        f"Error: saved response '{saved_response}' for {node.name} is invalid. "
                        "The remaining saved responses will not be applied, but you can continue from here."
                    )
                )
                return node

            node = node.next()
            q_and_a = next(q_and_a_iter, None)

        if q_and_a is not None:
            assert node is None
            rich_print(
                "Error: saved responses left to apply but no more questions in the tour. Aborting."
            )
            sys.exit(1)

        if node is not None:
            assert q_and_a is None
            rich_print(
                Panel(
                    "All saved responses were applied successfully, resuming where you left off."
                )
            )

        return node

    def run(self, resume_from: Optional[Path] = None):
        """Run the tour by traversing the tree depth-first"""
        if resume_from is not None:
            node = self.resume(resume_from)
        else:
            node = self.root
        while node is not None:
            if self.debug_state and node.name != "Root":
                rich_print(self.state)
            if self.trace and node.name != "Root":
                self.visualize(node)
            try:
                node.run()
            except KeyboardInterrupt:
                rich_print("\nKeyboard Interrupt")
                node = self.keyboard_interrupt_action(node)
                continue
            node = node.next()

    def visualize(self, highlight: Optional[Step] = None):
        """Display the tree structure of the tour on stdout"""

        def display(pre: str, name: str) -> str:
            return pre + name.replace(" Step", "").replace(
                "Representation", "Rep."
            ).replace("Root", "Wizard Steps")

        just_width = 4 + max(
            len(display(pre, node.name)) for pre, _, node in RenderTree(self.root)
        )
        text = ""
        for pre, _, node in RenderTree(self.root):
            treestr = display(pre, node.name)
            if highlight is not None:
                if node == highlight:
                    treestr = (
                        "[yellow]" + treestr.ljust(just_width) + "←———" + "[/yellow]"
                    )
                elif node.response is not None:
                    treestr = (
                        "[green]"
                        + (treestr + ":").ljust(just_width)
                        + str(node.response)
                        + "[/green]"
                    )
            text += treestr + "\n"
        rich_print(Panel(text.rstrip()))

    def get_progress(self, current_node: Step):
        """Return a list of questions and answers for the tour"""
        q_and_a_list = [[node.name, node.response] for node in PreOrderIter(self.root)]
        current_node_index = q_and_a_list.index(
            [current_node.name, current_node.response]
        )
        return q_and_a_list[:current_node_index]

    def save_progress(self, current_node: Step):
        """Save the questions and answers of the tour to a file for future resuming"""
        filename = questionary.path(
            "Where should we save your progress to?",
            default="",
            style=CUSTOM_QUESTIONARY_STYLE,
        ).ask()
        if not filename:
            rich_print("No output file provided, progress not saved.")
            return
        filename = sanitize_paths(filename)
        try:
            with open(filename, "w", encoding="utf8") as f:
                yaml.dump(
                    [["EveryVoice Wizard", VERSION]] + self.get_progress(current_node),
                    f,
                    allow_unicode=True,
                )
            rich_print(
                f"Saved progress to '{filename}'\n"
                f"You can resume from this state by running 'everyvoice new-project --resume-from {filename}'."
            )

            with open(filename, "r", encoding="utf8") as f:
                rich_print(yaml.safe_load(f))
        except OSError as e:
            rich_print(f"Error saving progress to {filename}: {e}")


class StepNames(Enum):
    name_step = "Name Step"
    contact_name_step = "Contact Name Step"
    contact_email_step = "Contact Email Step"
    dataset_name_step = "Dataset Name Step"
    dataset_permission_step = "Dataset Permission Step"
    output_step = "Output Path Step"
    wavs_dir_step = "Wavs Dir Step"
    filelist_step = "Filelist Step"
    filelist_format_step = "Filelist Format Step"
    validate_wavs_step = "Validate Wavs Step"
    filelist_text_representation_step = "Filelist Text Representation Step"
    target_training_representation_step = (
        "Target Training Representation Step"  # TODO: maybe don't need
    )
    data_has_header_line_step = "Filelist Has Header Line Step"
    basename_header_step = "Basename Header Step"
    text_header_step = "Text Header Step"
    data_has_speaker_value_step = "Data Has Speaker Step"
    speaker_header_step = "Speaker Header Step"
    know_speaker_step = "Know Speaker Step"
    add_speaker_step = "Add Speaker Step"
    data_has_language_value_step = "Data Has Language Step"
    language_header_step = "Language Header Step"
    select_language_step = "Select Language Step"
    text_processing_step = "Text Processing Step"
    g2p_step = "G2P Step"
    symbol_set_step = "Symbol-Set Step"
    sample_rate_config_step = "Sample Rate Config Step"
    audio_config_step = "Audio Config Step"
    sox_effects_step = "SoX Effects Step"
    more_datasets_step = "More Datasets Step"
    config_format_step = "Config Format Step"
