from enum import Enum
from typing import List, Union

from anytree import NodeMixin, RenderTree
from questionary import Style

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


class _Step:
    """The main class for steps within a tour.

    Each step must implement the prompt and validate methods.
    This method must save the answer in the response attribute.
    """

    def __init__(self):
        self.response = None
        self.completed = False

    def prompt(self):
        """Prompt the user and return the result.

        Raises:
            NotImplementedError: If you don't implement anything, this will be raised
        """
        raise NotImplementedError(
            f"This step ({self.name}) doesn't have a prompt method implemented. Please implement one."
        )

    def validate(self, response):
        """Validate the response.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError(
            f"This step ({self.name}) doesn't have a validate method implemented. Please implement one."
        )

    def effect(self):
        """Run some arbitrary code after the step resolves"""
        pass


class Step(_Step, NodeMixin):
    """Just a mixin to allow a tree-based interpretation of steps"""

    def __init__(
        self,
        name: Union[Enum, str],
        default=None,
        prompt_method=None,
        validate_method=None,
        effect_method=None,
        parent=None,
        children=None,
        state_subset=None,
    ):
        super(Step, self).__init__()
        self.name = name.value if isinstance(name, Enum) else name
        self.default = default
        self.parent = parent
        self.state_subset = state_subset
        self.state = None  # should be added when the Step is added to a Tour
        self.tour = None  # should be added when the Step is added to a Tour
        if effect_method is not None:
            self.effect = effect_method
        if prompt_method is not None:
            self.prompt = prompt_method
        if validate_method is not None:
            self.validate = validate_method
        if children:
            self.children = children

    def __repr__(self) -> str:
        return f"{self.name}: {super().__repr__()}"

    def run(self):
        """Prompt the user and save the response to the response attribute.
        If this method returns something truthy, continue, otherwise ask the prompt again.
        """
        self.response = self.prompt()
        if self.validate(self.response):
            self.completed = True
            try:
                if self.state is not None:
                    self.state[self.name] = self.response
                return self.response
            finally:
                self.effect()
        else:
            self.run()


class Tour:
    def __init__(self, name: str, steps: List[Step], state: dict = None):
        """Create the tour by setting each Step as the child of the previous Step."""
        self.name = name
        self.state = state if state is not None else {}
        for parent, child in zip(steps, steps[1:]):
            child.parent = parent
            self.determine_state(child, self.state)
            child.tour = self
        self.steps = steps
        self.root = steps[0]
        self.root.tour = self
        self.determine_state(self.root, self.state)

    def determine_state(self, step: Step, state: dict):
        if step.state_subset is not None:
            if step.state_subset not in state:
                state[step.state_subset] = {}
            step.state = state[step.state_subset]
        else:
            step.state = state

    def add_step(self, step: Step, parent: Step, child_index=0):
        self.determine_state(step, self.state)
        step.tour = self
        children = list(parent.children)
        children.insert(child_index, step)
        parent.children = children

    def run(self):
        for _, _, node in RenderTree(self.root):
            node.run()

    def visualize(self):
        for pre, _, node in RenderTree(self.root):
            treestr = f"{pre}{node.name}"
            print(treestr.ljust(8))


class StepNames(Enum):
    name_step = "Name Step"
    dataset_name_step = "Dataset Name Step"
    output_step = "Output Path Step"
    wavs_dir_step = "Wavs Dir Step"
    filelist_step = "Filelist Step"
    filelist_format_step = "Filelist Format Step"
    basename_header_step = "Basename Header Step"
    text_header_step = "Text Header Step"
    data_has_speaker_value_step = "Data Has Speaker Step"
    speaker_header_step = "Speaker Header Step"
    data_has_language_value_step = "Data Has Language Step"
    language_header_step = "Language Header Step"
    select_language_step = "Select Language Step"
    text_processing_step = "Text Processing Step"
    g2p_step = "G2P Step"
    symbol_set_step = "Symbol-Set step"
    sample_rate_config_step = "Sample Rate Config Step"
    audio_config_step = "Audio Config Step"
    sox_effects_step = "SoX Effects Step"
    more_datasets_step = "More Datasets Step"
    config_format_step = "Config Format Step"
