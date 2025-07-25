"""Unit tests for the wizard module"""

import os
import re
import string
import sys
import tempfile
from copy import deepcopy
from enum import Enum
from pathlib import Path
from textwrap import dedent
from types import MethodType
from typing import Callable, Iterable, NamedTuple, Optional, Sequence
from unittest import TestCase

from anytree import PreOrderIter, RenderTree
from packaging.version import Version

# [Unit testing questionary](https://github.com/prompt-toolkit/python-prompt-toolkit/blob/master/docs/pages/advanced_topics/unit_testing.rst)
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from everyvoice._version import VERSION
from everyvoice.tests.stubs import (
    Say,
    capture_stderr,
    capture_stdout,
    monkeypatch,
    null_patch,
    patch_input,
    patch_menu_prompt,
    patch_questionary,
    silence_c_stderr,
)
from everyvoice.text.phonemizer import AVAILABLE_G2P_ENGINES
from everyvoice.wizard import StepNames as SN
from everyvoice.wizard import basic, dataset, prompts, utils
from everyvoice.wizard.main_tour import get_main_wizard_tour
from everyvoice.wizard.tour import State, Step, Tour

CONTACT_INFO_STATE = State()
CONTACT_INFO_STATE[SN.contact_name_step.value] = "Test Name"
CONTACT_INFO_STATE[SN.contact_email_step.value] = "test@this.ca"


class RecursiveAnswers(NamedTuple):
    """Recursive answer for StepAndAnswer.children_answers, see StepAndAnswer
    documentation for a description of the fields here."""

    answer_or_monkey: Say | Callable
    children_answers: Optional[list["RecursiveAnswers"]] = None


class StepAndAnswer(NamedTuple):
    """named tuples to group together a step, its answer, and the answers of its children
    Members:
        step: an instance of a subclass of Step
        answer_or_monkey: either:
            - an instance of Say to get patched in for Step's prompt method
            or
            - a pre-instantiated monkeypatch context to use
        children_answers: an optional list of RecursiveAnswers to be used for
            the children of step this must align with what step.effect() will
            add as children given answer_or_monkey
    """

    step: Step
    answer_or_monkey: Say | Callable
    children_answers: Optional[list[RecursiveAnswers]] = None

    @property
    def monkey(self):
        """Return the monkey to use as a context manager"""
        if isinstance(self.answer_or_monkey, Say):
            return monkeypatch(self.step, "prompt", self.answer_or_monkey)
        else:
            return self.answer_or_monkey


def find_step(name: Enum, steps: Sequence[Step | list[Step]]):
    """Find a step with the given name in steps, of potentially variable depth"""
    for s in steps:
        if isinstance(s, list):
            try:
                return find_step(name, s)
            except IndexError:
                pass
        elif s.name == name.value:
            return s
    raise IndexError(f"Step {name} not found.")  # pragma: no cover


class TestStep(Step):
    """A Step subclass that allows specified the effect, prompt and validate
    methods in the constructor."""

    def __init__(
        self,
        name: str,
        parent=None,
        prompt_method: Optional[Callable] = None,
        validate_method: Optional[Callable] = None,
        effect_method: Optional[Callable] = None,
    ):
        super().__init__(name, parent=parent)
        if prompt_method:
            self.prompt = prompt_method  # type: ignore[method-assign]
        if validate_method:
            self.validate = validate_method  # type: ignore[method-assign]
        if effect_method:
            self.effect = effect_method  # type: ignore[method-assign]


def make_trivial_tour(trace: bool = False, debug_state: bool = False) -> Tour:
    return Tour(
        "trivial tour for testing",
        [
            basic.NameStep(),
            basic.ContactNameStep(),
            basic.ContactEmailStep(),
        ],
        trace=trace,
        debug_state=debug_state,
    )


class WizardTestBase(TestCase):
    """Base class for tests that involve the wizard and need to run tours."""

    def monkey_run_tour(
        self, name: str, steps_and_answers: list[StepAndAnswer], debug=False
    ) -> tuple[Tour, str]:
        """Create and run a tour with the monkey answers given

        Args:
            name: Name to give the tour when creating it
            steps_and_answers: a list of steps, answers, and optional recursive answers
                (Step, Answer/Monkey, optional [children answers/monkeys])
                where
                 - Step is an instantiated subclass of Step
                 - either:
                   - Answer is an instance of Say to get patched in for Step's prompt method
                   or
                   - Monkey is an instantiated monkeypatch context to use
                 - [children answers/monkeys] is an optional list of tuples
                   (Answer/Monkey, optional recursive [chidren answer/monkeys])
                   to be used recursively for the children of Step.
                   This must align with what Step.effect() adds as children.
            debug: if true, dump progress info while running steps

        Returns: (tour, logs_from_stdout)
        """

        def recursive_helper(steps_and_answers: Iterable[StepAndAnswer]):
            """Run all the steps with the patched answers, recursively running children
            steps that get added, if requested by the specification of children_answers.
            """
            for step_and_answer in steps_and_answers:
                step = step_and_answer.step
                # saved_monkey = copy(step_and_answer.monkey)
                print("Step:", step.name, file=sys.stderr)
                with step_and_answer.monkey:
                    step.run()

                # Test undoing and redoing every reversible step that passes through here.
                if step.is_reversible():
                    state = deepcopy(step.state)
                    step.undo()
                    self.assertNotEqual(state, step.state)
                    # with saved_monkey:
                    print(repr(step_and_answer.monkey), file=sys.stderr)
                    with step_and_answer.monkey:
                        step.run()
                    self.assertEqual(
                        state,
                        step.state,
                        f"Undo did not undo correctly for {step.name}",
                    )

                if step.children and step_and_answer.children_answers:
                    # Here we assemble steps_and_answers for the recursive call from
                    # the actual children of step and the provided children_answers.
                    recursive_helper(
                        steps_and_answers=[
                            StepAndAnswer(
                                child_step,
                                answer_or_monkey=recursive_answers.answer_or_monkey,
                                children_answers=recursive_answers.children_answers,
                            )
                            for child_step, recursive_answers in zip(
                                step.children,
                                step_and_answer.children_answers,
                            )
                        ]
                    )

        tour = Tour(name, steps=[step for (step, *_) in steps_and_answers])
        # fail on accidentally shared initializer
        self.assertTrue(tour.state == {} or tour.state == {"dataset_0": {}})
        with capture_stdout() as out:
            with null_patch() if debug else silence_c_stderr():
                recursive_helper(steps_and_answers)
        return tour, out.getvalue()


class WizardTest(WizardTestBase):
    """Basic test for the configuration wizard"""

    data_dir = Path(__file__).parent / "data"

    def test_implementation_missing(self):
        nothing_step = Step(name="Dummy Step")
        no_validate_step = TestStep(name="Dummy Step", prompt_method=lambda: "test")
        no_prompt_step = TestStep(name="Dummy Step", validate_method=lambda: True)
        for step in [nothing_step, no_validate_step, no_prompt_step]:
            with self.assertRaises(NotImplementedError):
                step.run()
            self.assertFalse(step.is_reversible())
            self.assertFalse(step.is_automatic())

    def test_config_format_effect(self):
        """This is testing if a null key can be passed without throwing an
        error, as reported by marc tessier.  There are no assertions, it is just
        testing that no exceptions get raised.
        """
        config_step = basic.ConfigFormatStep(name="Config Step")
        self.assertTrue(config_step.validate("yaml"))
        self.assertTrue(config_step.validate("json"))
        with tempfile.TemporaryDirectory() as tmpdirname:
            config_step.state = State()
            config_step.state[SN.output_step.value] = tmpdirname
            config_step.state[SN.name_step.value] = config_step.name
            config_step.state.update(CONTACT_INFO_STATE)
            config_step.state["dataset_test"] = State()
            config_step.state["dataset_test"][SN.symbol_set_step.value] = {
                "characters": list(string.ascii_letters)
            }
            config_step.state["dataset_test"][SN.wavs_dir_step.value] = (
                Path(tmpdirname) / "test"
            )
            config_step.state["dataset_test"][SN.dataset_name_step.value] = "test"
            config_step.state["dataset_test"]["filelist_data"] = [
                {
                    "basename": "0001",
                    "text": "hello",
                    "language": "und",
                    "speaker": "default",
                },
                {
                    "basename": "0002",
                    "text": "hello",
                    "language": "und",
                    "speaker": "default",
                    None: "test",
                },
                {
                    "basename": "0003.wav",
                    "text": "hello",
                    "language": "und",
                    "speaker": "default",
                    None: "test",
                },
            ]
            config_step.state["dataset_test"]["sox_effects"] = []
            with capture_stdout() as stdout:
                config_step.effect()
            with open(
                Path(tmpdirname) / "Config Step" / "test-filelist.psv", encoding="utf8"
            ) as f:
                self.assertEqual(
                    f.read(),
                    "basename|language|speaker|text\n0001|und|default|hello\n0002|und|default|hello\n0003|und|default|hello\n",
                )
            self.assertIn("Congratulations", stdout.getvalue())
            self.assertTrue(
                (Path(tmpdirname) / config_step.name / "logs_and_checkpoints").exists()
            )

    def test_access_response(self):
        root_step = TestStep(
            name="Dummy Step",
            prompt_method=lambda: "foo",
            validate_method=lambda x: True,
        )

        def validate(self, x):
            """Because the root step always returns 'foo', we only validate the second step if the prompt returns 'foobar'"""
            if self.parent.response + x == "foobar":
                return True

        second_step = TestStep(
            name="Dummy Step 2", prompt_method=lambda: "bar", parent=root_step
        )
        second_step.validate = MethodType(validate, second_step)
        for i, node in enumerate(PreOrderIter(root_step)):
            if i != 0:
                self.assertEqual(second_step.parent.response, "foo")
                self.assertTrue(node.validate("bar"))
                self.assertFalse(node.validate("foo"))
            node.run()

    def test_main_tour(self):
        tour = get_main_wizard_tour()
        self.assertGreater(len(tour.steps), 6)
        # TODO try to figure out how to actually run the tour in unit testing or
        # at least add more interesting assertions that just the fact that it's
        # got several steps.
        # self.monkey_run_tour() with a bunch of recursive answer would the thing to use here...

    def test_visualize(self):
        tour = get_main_wizard_tour()
        with capture_stdout() as out:
            tour.visualize()
        log = out.getvalue()
        self.assertIn("── Contact Name ", log)
        self.assertIn("── Validate Wavs ", log)

    def test_name_step(self):
        """Exercise providing a valid dataset name."""
        step = basic.NameStep()
        with capture_stdout() as stdout:
            with patch_input("myname"):
                step.run()
        self.assertEqual(step.response, "myname")
        self.assertIn("'myname'", stdout.getvalue())
        self.assertTrue(step.completed)

    def test_bad_name_step(self):
        """Exercise providing an invalid dataset name."""
        step = basic.NameStep("")
        # For unit testing, we cannot call step.run() if we patch a response
        # that will fail validation, because that would cause infinite
        # recursion.
        with capture_stdout() as stdout:
            self.assertFalse(step.validate("foo/bar"))
            self.assertFalse(step.validate(""))
        output = stdout.getvalue()
        self.assertIn("'foo/bar'", output)
        self.assertIn("is not valid", output)
        self.assertIn("your project needs a name", output)

        step = basic.NameStep("")
        with capture_stdout() as stdout:
            with patch_input(("bad/name", "good-name"), True):
                step.run()
        output = stdout.getvalue()
        self.assertIn("'bad/name'", stdout.getvalue())
        self.assertIn("is not valid", stdout.getvalue())
        self.assertEqual(step.response, "good-name")

    def test_bad_contact_name_step(self):
        """Exercise providing an invalid contact name."""
        step = basic.ContactNameStep("")
        with capture_stdout() as stdout:
            self.assertFalse(step.validate("a"))
            self.assertFalse(step.validate(""))
        output = stdout.getvalue()
        self.assertIn("Sorry", output)
        self.assertIn("EveryVoice requires a name", output)

    def test_bad_contact_email_step(self):
        """Exercise providing an invalid contact email."""
        step = basic.ContactEmailStep("")
        with capture_stdout() as stdout:
            self.assertFalse(step.validate("test"))
            self.assertFalse(step.validate("test@"))
            self.assertFalse(step.validate("test@test."))
            self.assertTrue(step.validate("test@test.ca"))
            self.assertFalse(step.validate(""))
        output = stdout.getvalue().replace(" \n", " ")
        # Supporting email-validator prior and post 2.2.0 where the error string changed.
        self.assertTrue(
            "It must have exactly one @-sign" in output
            or "An email address must have an @-sign" in output
        )
        self.assertIn("There must be something after the @-sign", output)
        self.assertIn("An email address cannot end with a period", output)

    def test_no_permissions(self):
        """Exercise lacking permissions, then trying again"""
        tour = get_main_wizard_tour()
        permission_step = find_step(SN.dataset_permission_step, tour.steps)
        self.assertGreater(len(permission_step.children), 8)
        self.assertGreater(len(tour.root.descendants), 14)
        self.assertIn("dataset_0", tour.state)
        with patch_menu_prompt(0):  # 0 is no, I don't have permission
            permission_step.run()
        self.assertEqual(permission_step.children, ())
        self.assertLess(len(tour.root.descendants), 10)
        self.assertNotIn("dataset_0", tour.state)

        more_dataset_step = find_step(SN.more_datasets_step, tour.steps)
        with patch_menu_prompt(1):  # 1 is Yes, I have more data
            more_dataset_step.run()
        self.assertIn("dataset_0", tour.state)
        self.assertGreater(len(more_dataset_step.descendants), 8)
        self.assertGreater(len(tour.root.descendants), 14)

    def test_output_path_step(self):
        """Exercise the OutputPathStep"""
        tour = Tour(
            "testing",
            [
                basic.NameStep(),
                basic.OutputPathStep(),
            ],
        )

        # We need to answer the name step before we can validate the output path step
        step = tour.steps[0]
        with capture_stdout():
            with patch_input("myname"):
                step.run()

        step = tour.steps[1]
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, "exits-as-file")
            # Bad case 1: output dir exists and is a file
            with open(file_path, "w", encoding="utf8") as f:
                f.write("blah")
            with capture_stdout():
                self.assertFalse(step.validate(file_path))

            # Bad case 2: file called the same as the dataset exists in the output dir
            dataset_file = os.path.join(tmpdirname, "myname")
            with open(dataset_file, "w", encoding="utf8") as f:
                f.write("blah")
            with capture_stdout():
                self.assertFalse(step.validate(tmpdirname))
            os.unlink(dataset_file)

            tmpdir = Path(tmpdirname)
            # Bad case 3: file under read-only directory
            if os.name != "nt":  # Windows does not support chmod
                ro_dir = tmpdir / "read-only"
                ro_dir.mkdir(mode=0x555)
                with capture_stdout() as out:
                    self.assertFalse(step.validate(str(ro_dir)))
                self.assertIn("could not create", out.getvalue())

            # Case with a deep path, make sure we don't leave part of it around
            self.assertTrue(step.validate(tmpdir / "deep" / "path"))
            self.assertFalse((tmpdir / "deep").exists())

            # Good case
            with capture_stdout() as stdout:
                with monkeypatch(step, "prompt", Say(tmpdirname)):
                    step.run()
            self.assertIn("will put your files", stdout.getvalue())

    def test_more_data_step(self):
        """Exercise giving an invalid response and a yes response to more data."""
        tour = Tour(
            "testing",
            [dataset.FilelistStep(state_subset="dataset_0"), basic.MoreDatasetsStep()],
        )

        step = tour.steps[1]
        self.assertFalse(step.validate("foo"))
        self.assertTrue(step.validate("yes"))
        self.assertEqual(len(step.children), 0)

        with patch_menu_prompt(0):  # answer 0 is "no"
            step.run()
        self.assertEqual(len(step.children), 1)
        self.assertIsInstance(step.children[0], basic.ConfigFormatStep)

        with patch_menu_prompt(1):  # answer 1 is "yes"
            step.run()
        self.assertGreater(len(step.descendants), 10)

    def test_no_data_to_save(self):
        """When the tour created no datasets at all, saving the config is skipped."""
        tour = Tour("testing", [basic.MoreDatasetsStep()])
        step = tour.steps[0]
        with patch_menu_prompt(0), capture_stdout() as out:  # answer 0 is "no"
            step.run()
        self.assertEqual(len(step.children), 0)
        self.assertIn("No dataset to save", out.getvalue())

    def test_dataset_name(self):
        step = dataset.DatasetNameStep()
        with patch_input(("", "bad/name", "good-name"), True):
            with capture_stdout() as stdout:
                step.run()
        output = stdout.getvalue().replace(" \n", " ").split("\n")
        self.assertIn("your dataset needs a name", output[0])
        self.assertIn("is not valid", output[1])
        self.assertIn("finished the configuration", "".join(output[2:]))
        self.assertTrue(step.completed)

    def test_speaker_name(self):
        step = dataset.AddSpeakerStep()
        with patch_input(("", "bad/name", "good-name"), True):
            with capture_stdout() as stdout:
                step.run()
        output = stdout.getvalue().replace(" \n", " ").split("\n")
        self.assertIn("speaker needs an ID", output[0])
        self.assertIn("is not valid", output[1])
        self.assertIn("will be used as the speaker ID", "".join(output[2:]))
        self.assertTrue(step.completed)

    def test_wavs_dir(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # our symlinks below aren't robust to relative paths...
            tmpdirname = os.path.abspath(tmpdirname)
            no_wavs_dir = os.path.join(tmpdirname, "no-wavs-here")
            os.mkdir(no_wavs_dir)

            has_wavs_dir = os.path.join(tmpdirname, "there-are-wavs-here")
            os.mkdir(has_wavs_dir)
            with open(os.path.join(has_wavs_dir, "foo.wav"), "wb") as f:
                f.write(b"A fantastic sounding clip! (or not...)")

            step = dataset.WavsDirStep("")
            with patch_questionary(("not-a-path", no_wavs_dir, has_wavs_dir)):
                with capture_stdout():
                    step.run()
            self.assertTrue(step.completed)
            self.assertEqual(step.response, has_wavs_dir)

            # No symlinks on Windows, so just skip those test case subitems
            if os.name == "nt":
                return

            wavs_are_symlinks = os.path.join(tmpdirname, "wavs-are-links-here")
            os.mkdir(wavs_are_symlinks)
            os.symlink(
                os.path.join(has_wavs_dir, "foo.wav"),
                os.path.join(wavs_are_symlinks, "foo.wav"),
            )
            step = dataset.WavsDirStep("")
            with patch_questionary(wavs_are_symlinks):
                with capture_stdout():
                    step.run()
            self.assertTrue(step.completed)
            self.assertEqual(step.response, wavs_are_symlinks)

            wavs_dir_is_symlink = os.path.join(tmpdirname, "link-to-wavs-dir")
            os.symlink(wavs_are_symlinks, wavs_dir_is_symlink)
            step = dataset.WavsDirStep("")
            with patch_questionary(wavs_dir_is_symlink):
                with capture_stdout():
                    step.run()
            self.assertTrue(step.completed)
            self.assertEqual(step.response, wavs_dir_is_symlink)

            # wavs_are_symlinks and wavs_dir_is_symlink pass even if
            # WavsDirStep.validate() were to use Path.glob(), but deeper_links
            # fails in that case, passing only if it uses os.path.glob(), so
            # this final test case is important.
            deeper_links = os.path.join(tmpdirname, "deeper-links")
            os.mkdir(deeper_links)
            os.symlink(wavs_dir_is_symlink, os.path.join(deeper_links, "nested-link"))
            step = dataset.WavsDirStep("")
            with patch_questionary(deeper_links):
                with capture_stdout():
                    step.run()
            self.assertTrue(step.completed)
            self.assertEqual(step.response, deeper_links)

    def test_sample_rate_config(self):
        step = dataset.SampleRateConfigStep("")
        with patch_questionary(
            (
                "not an int",  # obvious not valid
                "",  # ditto
                3.1415,  # floats are also not allowed
                50,  # this is below the minimum 100 allowed
                512,  # yay, a good response!
                1024,  # won't get used becasue 512 is good.
            )
        ):
            with capture_stdout() as stdout:
                step.run()
        output = stdout.getvalue().replace(" \n", " ").split(".\n")
        for i in range(4):
            self.assertIn("not a valid sample rate", output[i])
        self.assertTrue(step.completed)
        self.assertEqual(step.response, 512)

    def test_whitespace_always_collapsed(self):
        tour = Tour("unit testing", steps=dataset.get_dataset_steps())
        filelist = str(self.data_dir / "unit-test-case1.psv")
        filelist_step = find_step(SN.filelist_step, tour.steps)
        monkey = monkeypatch(filelist_step, "prompt", Say(filelist))
        with monkey:
            filelist_step.run()

        permission_step = find_step(SN.dataset_permission_step, tour.steps)
        with patch_menu_prompt(1):  # 1 is "yes, I have permission"
            permission_step.run()
        self.assertTrue(
            permission_step.state[SN.dataset_permission_step].startswith("Yes")
        )

        format_step = find_step(SN.filelist_format_step, tour.steps)
        with patch_menu_prompt(0):  # 0 is "psv"
            format_step.run()
        step = format_step.children[0]
        with patch_menu_prompt(1):  # 1 is "yes"
            step.run()

        step = format_step.children[1]
        with patch_menu_prompt(1):  # 1 is second column
            step.run()

        step = format_step.children[2]
        with patch_menu_prompt(1):  # 1 is second remaining column, i.e., third column
            step.run()

        text_representation_step = find_step(
            SN.filelist_text_representation_step, tour.steps
        )
        with patch_menu_prompt(0):  # 0 is "characters"
            text_representation_step.run()
        speaker_step = find_step(SN.data_has_speaker_value_step, tour.steps)
        with patch_menu_prompt(0):  # 0 is "no"
            speaker_step.run()

        know_speaker_step = speaker_step.children[0]
        with patch_menu_prompt(1):  # 1 is "yes"
            know_speaker_step.run()

        add_speaker_step = know_speaker_step.children[0]
        with patch_input("default"), capture_stdout():
            add_speaker_step.run()

        language_step = find_step(SN.data_has_language_value_step, tour.steps)
        with patch_menu_prompt(0):  # 0 is "no"
            language_step.run()

        select_lang_step = language_step.children[0]
        with capture_stdout(), capture_stderr():
            with patch_menu_prompt(15):  # some arbitrary language from the list
                select_lang_step.run()

        wavs_dir_step = find_step(SN.wavs_dir_step, tour.steps)
        with monkeypatch(wavs_dir_step, "prompt", Say(str(self.data_dir))):
            wavs_dir_step.run()

        validate_wavs_step = find_step(SN.validate_wavs_step, tour.steps)
        with patch_menu_prompt(1), capture_stdout():
            validate_wavs_step.run()

        text_processing_step = find_step(SN.text_processing_step, tour.steps)
        # 0 is lowercase, 1 is NFC Normalization, select none
        with monkeypatch(dataset, "tqdm", lambda seq, desc: seq):
            with patch_menu_prompt(None):
                text_processing_step.run()
        self.assertEqual(
            text_processing_step.state["filelist_data_list"][3][2],
            "let us see if it collapses whitespace",
        )

    def test_dataset_subtour(self):
        tour = Tour("unit testing", steps=dataset.get_dataset_steps())

        filelist = str(self.data_dir / "unit-test-case1.psv")
        filelist_step = find_step(SN.filelist_step, tour.steps)
        monkey = monkeypatch(filelist_step, "prompt", Say(filelist))
        with monkey:
            filelist_step.run()

        permission_step = find_step(SN.dataset_permission_step, tour.steps)
        with patch_menu_prompt(1):  # 1 is "yes, I have permission"
            permission_step.run()
        self.assertTrue(
            permission_step.state[SN.dataset_permission_step].startswith("Yes")
        )

        format_step = find_step(SN.filelist_format_step, tour.steps)
        with patch_menu_prompt(0):  # 0 is "psv"
            format_step.run()
        self.assertEqual(len(format_step.children), 3)

        step = format_step.children[0]
        self.assertIsInstance(step, dataset.HasHeaderLineStep)
        self.assertEqual(step.name, SN.data_has_header_line_step.value)
        with patch_menu_prompt(1):  # 1 is "yes"
            step.run()
        self.assertEqual(step.state[SN.data_has_header_line_step.value], "yes")
        self.assertEqual(len(step.state["filelist_data_list"]), 5)

        step = format_step.children[1]
        self.assertIsInstance(step, dataset.HeaderStep)
        self.assertEqual(step.name, SN.basename_header_step.value)
        with patch_menu_prompt(1):  # 1 is second column
            step.run()
        self.assertEqual(step.response, 1)
        self.assertEqual(step.state["filelist_headers"][1], "basename")

        step = format_step.children[2]
        self.assertIsInstance(step, dataset.HeaderStep)
        self.assertEqual(step.name, SN.text_header_step.value)
        with patch_menu_prompt(1):  # 1 is second remaining column, i.e., third column
            step.run()
        self.assertEqual(step.state["filelist_headers"][2], "text")

        text_representation_step = find_step(
            SN.filelist_text_representation_step, tour.steps
        )
        with patch_menu_prompt(0):  # 0 is "characters"
            text_representation_step.run()
        self.assertEqual(step.state["filelist_headers"][2], "characters")

        text_processing_step = find_step(SN.text_processing_step, tour.steps)
        # 0 is lowercase, 1 is NFC Normalization, select both
        with monkeypatch(dataset, "tqdm", lambda seq, desc: seq):
            with patch_menu_prompt([0, 1]):
                text_processing_step.run()
        # print(text_processing_step.state)
        self.assertEqual(
            text_processing_step.state["filelist_data_list"][2][2],
            "cased nfd: éàê nfc: éàê",  # the "nfd: éàê" bit here is now NFC
        )

        self.assertEqual(
            text_processing_step.state["filelist_data_list"][3][2],
            "let us see if it collapses whitespace",
        )

        speaker_step = find_step(SN.data_has_speaker_value_step, tour.steps)
        children_before = len(speaker_step.children)
        with patch_menu_prompt(0):  # 0 is "no"
            speaker_step.run()
        self.assertEqual(len(speaker_step.children), children_before + 1)
        self.assertIsInstance(speaker_step.children[0], dataset.KnowSpeakerStep)

        know_speaker_step = speaker_step.children[0]
        children_before = len(know_speaker_step.children)
        with patch_menu_prompt(1):  # 1 is "yes"
            know_speaker_step.run()
        self.assertEqual(len(know_speaker_step.children), children_before + 1)
        self.assertIsInstance(know_speaker_step.children[0], dataset.AddSpeakerStep)

        add_speaker_step = know_speaker_step.children[0]
        children_before = len(add_speaker_step.children)
        with patch_input("default"), capture_stdout():
            add_speaker_step.run()
        self.assertEqual(len(add_speaker_step.children), children_before)

        language_step = find_step(SN.data_has_language_value_step, tour.steps)
        children_before = len(language_step.children)
        with patch_menu_prompt(0):  # 0 is "no"
            language_step.run()
        self.assertEqual(len(language_step.children), children_before + 1)
        self.assertIsInstance(language_step.children[0], dataset.SelectLanguageStep)

        select_lang_step = language_step.children[0]
        with capture_stdout(), capture_stderr():
            # We rely on Finnish g2p in assertions below, so find it in the list.
            # 2 + index(fin) because in select_lang_step, [0] is "und", [1] is "custom",
            # and then follow the contents of AVAILABLE_G2P_ENGINES in order.
            fin_index = 2 + list(AVAILABLE_G2P_ENGINES).index("fin")
            with patch_menu_prompt(fin_index):
                select_lang_step.run()
        # print(select_lang_step.state)
        self.assertEqual(
            select_lang_step.state["filelist_headers"],
            ["unknown_0", "basename", "characters", "unknown_3"],
        )

        # Make sure realoading the data as dict stripped the header line
        self.assertEqual(len(step.state["filelist_data"]), 4)

        custom_g2p_step = find_step(SN.custom_g2p_step, tour.steps)
        with monkeypatch(custom_g2p_step, "prompt", Say(0)):
            custom_g2p_step.run()
        self.assertEqual(custom_g2p_step.language_codes, ["fin"])

        wavs_dir_step = find_step(SN.wavs_dir_step, tour.steps)
        with monkeypatch(wavs_dir_step, "prompt", Say(str(self.data_dir))):
            wavs_dir_step.run()

        validate_wavs_step = find_step(SN.validate_wavs_step, tour.steps)
        with patch_menu_prompt(1), capture_stdout() as out:
            validate_wavs_step.run()
        self.assertEqual(step.state[SN.validate_wavs_step][:2], "No")
        self.assertRegex(out.getvalue(), "Warning: .*4.* wav files were not found")

        symbol_set_step = find_step(SN.symbol_set_step, tour.steps)
        self.assertEqual(len(symbol_set_step.state["filelist_data"]), 4)
        with capture_stdout(), capture_stderr():
            symbol_set_step.run()
        self.assertEqual(len(symbol_set_step.state[SN.symbol_set_step.value]), 2)
        self.assertIn("t͡s", symbol_set_step.state[SN.symbol_set_step.value]["phones"])
        self.assertNotIn(
            ":", symbol_set_step.state[SN.symbol_set_step.value]["characters"]
        )
        self.assertNotIn(":", symbol_set_step.state[SN.symbol_set_step.value]["phones"])
        # assert that symbols contain no duplicates
        self.assertEqual(
            len(set(symbol_set_step.state[SN.symbol_set_step.value]["characters"])),
            len(symbol_set_step.state[SN.symbol_set_step.value]["characters"]),
        )
        self.assertEqual(
            len(set(symbol_set_step.state[SN.symbol_set_step.value]["phones"])),
            len(symbol_set_step.state[SN.symbol_set_step.value]["phones"]),
        )

        sox_effects_step = find_step(SN.sox_effects_step, tour.steps)
        # 0 is resample to 22050 kHz, 2 is remove silence at start and end
        with patch_menu_prompt([0, 1]):
            sox_effects_step.run()
        # print(sox_effects_step.state["sox_effects"])
        self.assertEqual(
            sox_effects_step.state["sox_effects"],
            [
                ["channels", "1"],
                ["norm", "-3.0"],
                [
                    "silence",
                    "1",
                    "0.1",
                    "0.1%",
                ],
                ["reverse"],
                [
                    "silence",
                    "1",
                    "0.1",
                    "0.1%",
                ],
                ["reverse"],
            ],
        )

    def test_empty_filelist(self):
        tour = Tour(
            name="empty filelist",
            steps=[
                dataset.FilelistStep(),
                dataset.FilelistFormatStep(),
            ],
        )
        filelist = str(self.data_dir / "empty.psv")

        filelist_step = tour.steps[0]
        with monkeypatch(filelist_step, "prompt", Say(filelist)):
            filelist_step.run()

        format_step = tour.steps[1]
        with self.assertRaises(SystemExit) as cm:
            with patch_menu_prompt(1) as stdout:
                format_step.run()
            output = stdout.getvalue()
            self.assertIn("is empty", output)
        self.assertEqual(cm.exception.code, 1)

    def test_wrong_fileformat_psv(self):
        tour = Tour(
            name="mismatched fileformat",
            steps=[
                dataset.FilelistStep(),
                dataset.FilelistFormatStep(),
            ],
        )
        filelist = str(self.data_dir / "unit-test-case2.psv")

        filelist_step = tour.steps[0]
        with monkeypatch(filelist_step, "prompt", Say(filelist)):
            filelist_step.run()
        format_step = tour.steps[1]
        # try with: 1/tsv (wrong), 2/csv (wrong), 3/festival (wrong) and finally 0 tsv (right)
        with patch_menu_prompt((1, 2, 3, 0), multi=True) as stdout:
            format_step.run()
        output = re.sub(r" *\n", " ", stdout.getvalue())
        self.assertRegex(output, r"does not look like a .*'tsv'".replace(" ", r"\s"))
        self.assertRegex(output, r"does not look like a .*'csv'".replace(" ", r"\s"))
        self.assertIn("is not in the festival format", output)
        self.assertTrue(format_step.completed)
        # print(format_step.state)

    def test_wrong_fileformat_festival(self):
        tour = Tour(
            name="mismatched fileformat",
            steps=[
                dataset.FilelistStep(),
                dataset.FilelistFormatStep(),
            ],
        )
        filelist = str(self.data_dir / "unit-test-case3.festival")

        filelist_step = tour.steps[0]
        with monkeypatch(filelist_step, "prompt", Say(filelist)):
            filelist_step.run()
        format_step = tour.steps[1]
        # try with: 0/psv (wrong), 1/tsv (wrong), 2/csv (wrong), and finally 3/festival (right)
        with patch_menu_prompt((0, 1, 2, 3), multi=True) as stdout:
            format_step.run()
        output = stdout.getvalue().replace(" \n", " ")
        self.assertRegex(output, r"does not look like a .*'psv'".replace(" ", r"\s"))
        self.assertRegex(output, r"does not look like a .*'tsv'".replace(" ", r"\s"))
        self.assertRegex(output, r"does not look like a .*'csv'".replace(" ", r"\s"))
        self.assertTrue(format_step.completed)
        # print(format_step.state)

    def test_validate_path(self):
        """Unit testing for validate_path() in isolation."""
        from everyvoice.wizard.validators import validate_path

        with self.assertRaises(ValueError):
            validate_path("", is_dir=False, is_file=False)
        with self.assertRaises(ValueError):
            validate_path("", is_dir=True, is_file=True)
        with self.assertRaises(ValueError):
            validate_path("")
        with tempfile.TemporaryDirectory() as tmpdirname, capture_stdout():
            self.assertTrue(
                validate_path(tmpdirname, is_dir=True, is_file=False, exists=True)
            )
            self.assertFalse(
                validate_path(tmpdirname, is_dir=False, is_file=True, exists=True)
            )
            self.assertFalse(
                validate_path(tmpdirname, is_dir=True, is_file=False, exists=False)
            )
            file_name = os.path.join(tmpdirname, "some-file-name")
            with open(file_name, "w", encoding="utf8") as f:
                f.write("foo")
            self.assertTrue(
                validate_path(file_name, is_dir=False, is_file=True, exists=True)
            )
            self.assertFalse(
                validate_path(file_name, is_dir=False, is_file=True, exists=False)
            )
            self.assertFalse(
                validate_path(file_name, is_dir=True, is_file=False, exists=True)
            )

            not_file_name = os.path.join(tmpdirname, "file-does-not-exist")
            self.assertFalse(
                validate_path(not_file_name, is_dir=False, is_file=True, exists=True)
            )
            self.assertTrue(
                validate_path(not_file_name, is_dir=False, is_file=True, exists=False)
            )

    def test_prompt(self):
        with patch_menu_prompt(0):
            answer = prompts.get_response_from_menu_prompt(
                choices=("choice1", "choice2")
            )
            self.assertEqual(answer, "choice1")
        with patch_menu_prompt(1) as stdout:
            answer = prompts.get_response_from_menu_prompt(
                "some question", ("choice1", "choice2")
            )
            self.assertEqual(answer, "choice2")
            self.assertIn("some question", stdout.getvalue())
        with patch_menu_prompt((2, 4)):
            answer = prompts.get_response_from_menu_prompt(
                choices=("a", "b", "c", "d", "e"), multi=True
            )
            self.assertEqual(answer, ["c", "e"])
        with patch_menu_prompt(1):
            answer = prompts.get_response_from_menu_prompt(
                choices=("a", "b", "c", "d", "e"), return_indices=True
            )
            self.assertEqual(answer, 1)

    def test_monkey_tour_1(self):
        with tempfile.TemporaryDirectory() as tmpdirname, capture_stdout():
            tour, _ = self.monkey_run_tour(
                "monkey tour 1",
                [
                    StepAndAnswer(basic.NameStep(), patch_input("my-dataset-name")),
                    StepAndAnswer(
                        basic.OutputPathStep(), patch_questionary(tmpdirname)
                    ),
                ],
            )
        self.assertEqual(tour.state[SN.name_step.value], "my-dataset-name")
        self.assertEqual(tour.state[SN.output_step.value], tmpdirname)

    def test_monkey_tour_2(self):
        data_dir = Path(__file__).parent / "data"
        tour, out = self.monkey_run_tour(
            "monkey tour 2",
            [
                StepAndAnswer(
                    dataset.FilelistStep(),
                    patch_questionary(data_dir / "metadata.psv"),
                ),
                StepAndAnswer(
                    dataset.FilelistFormatStep(),
                    patch_menu_prompt(0),  # psv
                ),
                StepAndAnswer(
                    dataset.FilelistTextRepresentationStep(),
                    patch_menu_prompt(0),  # characters
                ),
                StepAndAnswer(
                    dataset.HasSpeakerStep(),
                    patch_menu_prompt(1),  # "yes"
                    children_answers=[RecursiveAnswers(Say(3))],
                ),
                StepAndAnswer(
                    dataset.HasLanguageStep(),
                    patch_menu_prompt(0),  # "no"
                    children_answers=[RecursiveAnswers(Say("eng"))],
                ),
                StepAndAnswer(dataset.WavsDirStep(), patch_questionary(data_dir)),
                StepAndAnswer(
                    dataset.ValidateWavsStep(),
                    patch_menu_prompt(0),  # 0 is Yes
                    children_answers=[
                        RecursiveAnswers(patch_questionary(data_dir / "lj/wavs")),
                        RecursiveAnswers(null_patch()),
                    ],
                ),
                StepAndAnswer(dataset.TextProcessingStep(), patch_menu_prompt([0, 1])),
                StepAndAnswer(dataset.SymbolSetStep(), null_patch()),
                StepAndAnswer(dataset.SoxEffectsStep(), patch_menu_prompt([0])),
                StepAndAnswer(
                    dataset.DatasetNameStep(), patch_input("my-monkey-dataset")
                ),
            ],
        )

        tree = str(RenderTree(tour.root))
        self.assertIn("├── Validate Wavs Step", tree)
        self.assertIn("│   └── Validate Wavs Step", tree)
        self.assertIn("Great! All audio files found in directory", out)

        # print(tour.state)
        self.assertEqual(len(tour.state["filelist_data"]), 5)
        self.assertTrue(tour.steps[-1].completed)

    def test_get_iso_code(self):
        self.assertEqual(utils.get_iso_code("eng"), "eng")
        self.assertEqual(utils.get_iso_code("[eng]"), "eng")
        self.assertEqual(utils.get_iso_code("es"), "es")
        self.assertEqual(utils.get_iso_code("[es]"), "es")
        self.assertIs(utils.get_iso_code(None), None)

    def test_with_language_column(self):
        data_dir = Path(__file__).parent / "data"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tour, _ = self.monkey_run_tour(
                "tour with language column",
                [
                    StepAndAnswer(basic.NameStep(), patch_input("project")),
                    StepAndAnswer(basic.ContactNameStep(), patch_input("Test Name")),
                    StepAndAnswer(
                        basic.ContactEmailStep(), patch_input("info@everyvoice.ca")
                    ),
                    StepAndAnswer(
                        basic.OutputPathStep(), patch_questionary(tmpdir / "out")
                    ),
                    StepAndAnswer(
                        dataset.WavsDirStep(state_subset="dataset_0"),
                        patch_questionary(data_dir),
                    ),
                    StepAndAnswer(
                        dataset.FilelistStep(state_subset="dataset_0"),
                        patch_questionary(data_dir / "language-col.tsv"),
                    ),
                    StepAndAnswer(
                        dataset.FilelistFormatStep(state_subset="dataset_0"),
                        patch_menu_prompt(1),  # tsv
                    ),
                    StepAndAnswer(
                        dataset.FilelistTextRepresentationStep(
                            state_subset="dataset_0"
                        ),
                        patch_menu_prompt(0),  # characters
                    ),
                    StepAndAnswer(
                        dataset.TextProcessingStep(state_subset="dataset_0"),
                        patch_menu_prompt([0, 1]),
                    ),
                    StepAndAnswer(
                        dataset.HasSpeakerStep(state_subset="dataset_0"),
                        patch_menu_prompt(1),  # "yes"
                        children_answers=[RecursiveAnswers(Say(2))],
                    ),
                    StepAndAnswer(
                        dataset.HasLanguageStep(state_subset="dataset_0"),
                        patch_menu_prompt(1),  # "yes"
                        children_answers=[RecursiveAnswers(Say(3))],
                    ),
                    StepAndAnswer(
                        dataset.SymbolSetStep(state_subset="dataset_0"),
                        null_patch(),
                    ),
                    StepAndAnswer(
                        dataset.SoxEffectsStep(state_subset="dataset_0"),
                        patch_menu_prompt([0]),
                    ),
                    StepAndAnswer(
                        dataset.DatasetNameStep(state_subset="dataset_0"),
                        patch_input("my-monkey-dataset"),
                    ),
                    StepAndAnswer(
                        basic.MoreDatasetsStep(),
                        patch_menu_prompt(0),
                        children_answers=[RecursiveAnswers(patch_menu_prompt(0))],
                    ),
                ],
            )

            self.assertEqual(tour.state["dataset_0"][SN.speaker_header_step.value], 2)
            self.assertEqual(tour.state["dataset_0"][SN.language_header_step.value], 3)
            self.assertTrue(tour.steps[-1].completed)

            with open(
                tmpdir / "out/project/config/everyvoice-text-to-spec.yaml",
                encoding="utf8",
            ) as f:
                text_to_spec_config = "\n".join(f)
            self.assertIn("multilingual: true", text_to_spec_config)
            self.assertIn("multispeaker: true", text_to_spec_config)

    def test_no_header_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with open(tmpdir / "filelist.psv", "w", encoding="utf8") as f:
                f.write("f1|foo bar|Joe\nf2|bar baz|Joe\nf3|baz foo|Joe\n")
            for basename in ("f1", "f2", "f3"):
                with open(tmpdir / (basename + ".wav"), "wb"):
                    pass
            tour, _ = self.monkey_run_tour(
                "Tour with datafile missing the header line",
                [
                    StepAndAnswer(basic.NameStep(), patch_input("project")),
                    StepAndAnswer(basic.ContactNameStep(), patch_input("Test Name")),
                    StepAndAnswer(
                        basic.ContactEmailStep(), patch_input("info@everyvoice.ca")
                    ),
                    StepAndAnswer(
                        basic.OutputPathStep(), patch_questionary(tmpdir / "out")
                    ),
                    StepAndAnswer(
                        dataset.WavsDirStep(state_subset="dataset_0"),
                        patch_questionary(tmpdir),
                    ),
                    StepAndAnswer(
                        dataset.FilelistStep(state_subset="dataset_0"),
                        patch_questionary(tmpdir / "filelist.psv"),
                    ),
                    StepAndAnswer(
                        dataset.FilelistFormatStep(state_subset="dataset_0"),
                        patch_menu_prompt(0),  # psv
                        children_answers=[
                            RecursiveAnswers(patch_menu_prompt(0)),  # no header line
                            RecursiveAnswers(Say(0)),  # column 0 is basename
                            RecursiveAnswers(Say(1)),  # column 1 is text
                        ],
                    ),
                    StepAndAnswer(
                        dataset.FilelistTextRepresentationStep(
                            state_subset="dataset_0"
                        ),
                        patch_menu_prompt(0),  # characters
                    ),
                    StepAndAnswer(
                        dataset.HasSpeakerStep(state_subset="dataset_0"),
                        patch_menu_prompt(0),  # "no"
                        children_answers=[
                            RecursiveAnswers(
                                patch_menu_prompt(0)
                            ),  # don't want to specify speaker ID
                        ],
                    ),
                    StepAndAnswer(
                        dataset.HasLanguageStep(state_subset="dataset_0"),
                        patch_menu_prompt(0),
                        children_answers=[RecursiveAnswers(patch_menu_prompt(0))],
                    ),
                    StepAndAnswer(
                        dataset.TextProcessingStep(state_subset="dataset_0"),
                        patch_menu_prompt(()),
                    ),
                    StepAndAnswer(
                        dataset.SymbolSetStep(state_subset="dataset_0"),
                        null_patch(),
                    ),
                    StepAndAnswer(
                        dataset.SoxEffectsStep(state_subset="dataset_0"),
                        patch_menu_prompt(None),
                    ),
                    StepAndAnswer(
                        dataset.DatasetNameStep(state_subset="dataset_0"),
                        patch_input("dataset"),
                    ),
                    StepAndAnswer(
                        basic.MoreDatasetsStep(),
                        patch_menu_prompt(0),
                        children_answers=[RecursiveAnswers(patch_menu_prompt(0))],
                    ),
                ],
            )
            self.assertEqual(len(tour.state["dataset_0"]["filelist_data"]), 3)
            with open(
                tmpdir / "out/project/dataset-filelist.psv", encoding="utf8"
            ) as f:
                output_filelist = list(f)
            self.assertEqual(len(output_filelist), 4)
            with open(
                tmpdir / "out/project/config/everyvoice-text-to-spec.yaml",
                encoding="utf8",
            ) as f:
                text_to_spec_config = "\n".join(f)
            self.assertIn("multilingual: false", text_to_spec_config)
            self.assertIn("multispeaker: false", text_to_spec_config)

    def test_running_out_of_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with open(tmpdir / "filelist.psv", "w", encoding="utf8") as f:
                f.write("basename|text\nf1|foo bar\nf2|bar baz\nf3|baz foo\n")
            for basename in ("f1", "f2", "f3"):
                with open(tmpdir / (basename + ".wav"), "wb"):
                    pass
            tour, _ = self.monkey_run_tour(
                "Tour without enough columns to have speaker or language",
                [
                    StepAndAnswer(basic.NameStep(), patch_input("project")),
                    StepAndAnswer(basic.ContactNameStep(), patch_input("Test Name")),
                    StepAndAnswer(
                        basic.ContactEmailStep(), patch_input("info@everyvoice.ca")
                    ),
                    StepAndAnswer(
                        basic.OutputPathStep(), patch_questionary(tmpdir / "out")
                    ),
                    StepAndAnswer(
                        dataset.WavsDirStep(state_subset="dataset_0"),
                        patch_questionary(tmpdir),
                    ),
                    StepAndAnswer(
                        dataset.FilelistStep(state_subset="dataset_0"),
                        patch_questionary(tmpdir / "filelist.psv"),
                    ),
                    StepAndAnswer(
                        dataset.FilelistFormatStep(state_subset="dataset_0"),
                        patch_menu_prompt(0),  # psv
                    ),
                    StepAndAnswer(
                        dataset.FilelistTextRepresentationStep(
                            state_subset="dataset_0"
                        ),
                        patch_menu_prompt(0),  # characters
                    ),
                    StepAndAnswer(
                        dataset.HasSpeakerStep(state_subset="dataset_0"),
                        null_patch(),  # user won't get prompted since there are no columns
                        children_answers=[
                            RecursiveAnswers(
                                patch_menu_prompt(0)
                            )  # don't want to specify speaker ID
                        ],
                    ),
                    StepAndAnswer(
                        dataset.HasLanguageStep(state_subset="dataset_0"),
                        null_patch(),  # user won't get prompted since there are no columns
                    ),
                    StepAndAnswer(
                        dataset.SelectLanguageStep(state_subset="dataset_0"),
                        patch_menu_prompt(0),
                    ),
                    StepAndAnswer(
                        dataset.TextProcessingStep(state_subset="dataset_0"),
                        patch_menu_prompt(()),
                    ),
                    StepAndAnswer(
                        dataset.SymbolSetStep(state_subset="dataset_0"),
                        null_patch(),
                    ),
                    StepAndAnswer(
                        dataset.SoxEffectsStep(state_subset="dataset_0"),
                        patch_menu_prompt([]),
                    ),
                    StepAndAnswer(
                        dataset.DatasetNameStep(state_subset="dataset_0"),
                        patch_input("dataset"),
                    ),
                    StepAndAnswer(
                        basic.MoreDatasetsStep(),
                        patch_menu_prompt(0),
                        children_answers=[RecursiveAnswers(patch_menu_prompt(0))],
                    ),
                ],
            )
            self.assertEqual(
                tour.state["dataset_0"]["filelist_headers"], ["basename", "characters"]
            )
            self.assertEqual(
                tour.state["dataset_0"][SN.data_has_speaker_value_step.value], "no"
            )
            self.assertEqual(
                tour.state["dataset_0"][SN.data_has_language_value_step.value], "no"
            )
            self.assertEqual(len(tour.state["dataset_0"]["filelist_data"]), 3)
            with open(
                tmpdir / "out/project/dataset-filelist.psv", encoding="utf8"
            ) as f:
                output_filelist = [line.rstrip() for line in f]
            expected_filelist = [
                "basename|language|speaker|characters|phones",
                "f1|und|speaker_0|foo bar|foo bar",
                "f2|und|speaker_0|bar baz|bar baz",
                "f3|und|speaker_0|baz foo|baz foo",
            ]
            self.assertListEqual(output_filelist, expected_filelist)

    def test_keyboard_interrupt(self):
        step = basic.NameStep()
        with self.assertRaises(KeyboardInterrupt), capture_stdout():
            with patch_input(KeyboardInterrupt()):
                step.run()

        step = dataset.WavsDirStep()
        with self.assertRaises(KeyboardInterrupt):
            with patch_questionary([KeyboardInterrupt()]):
                step.run()

        step = basic.MoreDatasetsStep()
        with self.assertRaises(KeyboardInterrupt):
            with patch_menu_prompt(KeyboardInterrupt()):
                step.run()

    def test_give_up_after_twenty_failures(self):
        step = dataset.WavsDirStep()
        with monkeypatch(step, "prompt", Say("no/such/directory")):
            with capture_stdout(), self.assertRaises(SystemExit):
                step.run()

    def test_leading_white_space_in_outpath(self):
        """
        Make sure we strip leading spaces when the user accidentally adds a
        leading space when specifying an output path.
        """
        with capture_stdout(), tempfile.TemporaryDirectory() as tmpdirname:
            with create_pipe_input() as pipe_input:
                # NOTE: we use `` to replace the `.` with our path with leading spaces.
                pipe_input.send_text(f"  {tmpdirname}\n")
                with create_app_session(input=pipe_input, output=DummyOutput()):
                    tour = Tour(
                        name="trimming leading spaces",
                        steps=[
                            basic.OutputPathStep(),
                        ],
                        state={SN.name_step.value: "output_dir_with_leading_spaces"},
                    )
                    tour.run()
        self.assertFalse(tour.state[SN.output_step.value].startswith(" "))
        self.assertEqual(tour.state[SN.output_step.value], tmpdirname)

    def test_leading_white_space_in_wav_dir(self):
        """
        Make sure we strip leading spaces when the user accidentally adds a
        leading space when specifying a wav directory.
        """
        step = dataset.WavsDirStep()
        path = Path(__file__).parent
        with create_pipe_input() as pipe_input:
            pipe_input.send_text(f" {path}\n")
            with create_app_session(input=pipe_input, output=DummyOutput()):
                step.run()
        self.assertFalse(step.response.startswith(" "))
        self.assertEqual(step.response, str(path))

    def test_leading_white_space_in_filelist(self):
        """
        Make sure we strip leading spaces when the user accidentally adds a
        leading space when specifying a filelist.
        """
        step = dataset.FilelistStep()
        path = Path(__file__).parent / "data/unit-test-case1.psv"
        with create_pipe_input() as pipe_input:
            pipe_input.send_text(f" {path}\n")
            with create_app_session(input=pipe_input, output=DummyOutput()):
                step.run()
        self.assertFalse(step.response.startswith(" "))
        self.assertEqual(step.response, str(path))

    def test_festival(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with open(tmpdir / "filelist.txt", "w", encoding="utf8") as f:
                f.write(
                    "\n".join(
                        [
                            '( f1 " foo bar " )',
                            '( f2 " bar baz " )',
                            '( f3 " baz foo " )',
                        ]
                    )
                )
            for basename in ("f1", "f2", "f3"):
                with open(tmpdir / (basename + ".wav"), "wb"):
                    pass
            tour, _ = self.monkey_run_tour(
                "Tour with datafile in the festival format",
                [
                    StepAndAnswer(basic.NameStep(), patch_input("project")),
                    StepAndAnswer(basic.ContactNameStep(), patch_input("Test Name")),
                    StepAndAnswer(
                        basic.ContactEmailStep(), patch_input("info@everyvoice.ca")
                    ),
                    StepAndAnswer(
                        basic.OutputPathStep(), patch_questionary(tmpdir / "out")
                    ),
                    StepAndAnswer(
                        dataset.WavsDirStep(state_subset="dataset_0"),
                        patch_questionary(tmpdir),
                    ),
                    StepAndAnswer(
                        dataset.FilelistStep(state_subset="dataset_0"),
                        patch_questionary(tmpdir / "filelist.txt"),
                    ),
                    StepAndAnswer(
                        dataset.FilelistFormatStep(state_subset="dataset_0"),
                        patch_menu_prompt(3),  # festival
                    ),
                    StepAndAnswer(
                        dataset.FilelistTextRepresentationStep(
                            state_subset="dataset_0"
                        ),
                        patch_menu_prompt(0),  # characters
                    ),
                    StepAndAnswer(
                        dataset.HasSpeakerStep(state_subset="dataset_0"),
                        null_patch(),
                        children_answers=[
                            RecursiveAnswers(
                                patch_menu_prompt(1),
                                children_answers=[
                                    RecursiveAnswers(patch_input("default_speaker"))
                                ],
                            ),  # want to specify speaker ID
                        ],
                    ),
                    StepAndAnswer(
                        dataset.HasLanguageStep(state_subset="dataset_0"),
                        null_patch(),
                        children_answers=[
                            RecursiveAnswers(
                                patch_menu_prompt(0),  # "und" lang selection
                            )
                        ],
                    ),
                    StepAndAnswer(
                        dataset.TextProcessingStep(state_subset="dataset_0"),
                        patch_menu_prompt((1,)),  # one arbitrary text processor
                    ),
                    StepAndAnswer(
                        dataset.SymbolSetStep(state_subset="dataset_0"),
                        null_patch(),
                    ),
                    StepAndAnswer(
                        dataset.SoxEffectsStep(state_subset="dataset_0"),
                        patch_menu_prompt([]),
                    ),
                    StepAndAnswer(
                        dataset.DatasetNameStep(state_subset="dataset_0"),
                        patch_input("dataset"),
                    ),
                    StepAndAnswer(
                        basic.MoreDatasetsStep(),
                        patch_menu_prompt(0),
                        children_answers=[RecursiveAnswers(patch_menu_prompt(0))],
                    ),
                ],
            )
            with open(
                tmpdir / "out/project/dataset-filelist.psv", encoding="utf8"
            ) as f:
                output_filelist = [line.rstrip() for line in f]
            expected_filelist = [
                "basename|language|speaker|characters|phones",
                "f1|und|default_speaker|foo bar|foo bar",
                "f2|und|default_speaker|bar baz|bar baz",
                "f3|und|default_speaker|baz foo|baz foo",
            ]
            self.assertListEqual(output_filelist, expected_filelist)

    def test_multilingual_multispeaker_true_config(self):
        """
        Test mismatched multi-monolingual and multi-monospeaker datasets.

        Makes sure that multilingual and multispeaker parameters of config are set to true when two monolingual and monospeaker datasets are provided with different specified languages and speakers.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with open(tmpdir / "filelist1.psv", "w", encoding="utf8") as f:
                f.write(
                    "\n".join(
                        [
                            "basename|language|speaker|text",
                            "f1|eng|speaker_1|foo foo",
                            "f2|eng|speaker_1|bar bar",
                            "f3|eng|speaker_1|baz baz",
                        ]
                    )
                )
            with open(tmpdir / "filelist2.txt", "w", encoding="utf8") as f:
                f.write(
                    "\n".join(
                        [
                            '( f4 " foo bar " )',
                            '( f5 " bar baz " )',
                            '( f6 " baz foo " )',
                        ]
                    )
                )
            for basename in ("f1", "f2", "f3", "f4", "f5", "f6"):
                with open(tmpdir / (basename + ".wav"), "wb"):
                    pass
            more_dataset_children_answers = [
                RecursiveAnswers(
                    patch_questionary(tmpdir / "filelist2.txt")
                ),  # filelist step
                RecursiveAnswers(
                    patch_menu_prompt(1),  # yes to permission
                    children_answers=[
                        RecursiveAnswers(patch_menu_prompt(3)),  # festival format
                        RecursiveAnswers(patch_menu_prompt(0)),  # characters
                        RecursiveAnswers(
                            patch_menu_prompt(())
                        ),  # no text preprocessing
                        RecursiveAnswers(
                            null_patch(),  # no speaker column Q for festival format
                            children_answers=[
                                RecursiveAnswers(
                                    patch_menu_prompt(1),  # will specify the speaker ID
                                    children_answers=[
                                        # new speaker ID
                                        RecursiveAnswers(patch_input("default_speaker"))
                                    ],
                                ),
                            ],
                        ),
                        RecursiveAnswers(
                            null_patch(),  # no language column Q for festival format
                            children_answers=[
                                RecursiveAnswers(
                                    patch_menu_prompt(0),  # "und" lang selection
                                )
                            ],
                        ),
                        RecursiveAnswers(patch_menu_prompt(0)),  # Keep g2p settings
                        RecursiveAnswers(patch_questionary(tmpdir)),  # wav directory
                        RecursiveAnswers(null_patch()),  # ValidateWavsStep
                        RecursiveAnswers(null_patch()),  # SymbolSetStep
                        RecursiveAnswers(patch_menu_prompt([])),  # no Sox
                        RecursiveAnswers(patch_input("dataset1")),  # Dataset name
                    ],
                ),
                RecursiveAnswers(
                    patch_menu_prompt(0),
                    children_answers=[RecursiveAnswers(patch_menu_prompt(0))],
                ),
            ]
            steps_and_answers = [
                StepAndAnswer(basic.NameStep(), patch_input("project")),
                StepAndAnswer(basic.ContactNameStep(), patch_input("Test Name")),
                StepAndAnswer(
                    basic.ContactEmailStep(), patch_input("info@everyvoice.ca")
                ),
                StepAndAnswer(
                    basic.OutputPathStep(), patch_questionary(tmpdir / "out")
                ),
                # First dataset
                StepAndAnswer(
                    dataset.WavsDirStep(state_subset="dataset_0"),
                    patch_questionary(tmpdir),
                ),
                StepAndAnswer(
                    dataset.FilelistStep(state_subset="dataset_0"),
                    patch_questionary(tmpdir / "filelist1.psv"),
                ),
                StepAndAnswer(
                    dataset.FilelistFormatStep(state_subset="dataset_0"),
                    patch_menu_prompt(0),  # psv
                ),
                StepAndAnswer(
                    dataset.FilelistTextRepresentationStep(state_subset="dataset_0"),
                    patch_menu_prompt(0),  # characters
                ),
                StepAndAnswer(
                    dataset.HasSpeakerStep(state_subset="dataset_0"),
                    patch_menu_prompt(1),  # 1 is yes
                    children_answers=[RecursiveAnswers(Say(2))],
                ),
                StepAndAnswer(
                    dataset.HasLanguageStep(state_subset="dataset_0"),
                    patch_menu_prompt(1),  # 1 is yes
                    children_answers=[RecursiveAnswers(Say(1))],
                ),
                StepAndAnswer(
                    dataset.TextProcessingStep(state_subset="dataset_0"),
                    patch_menu_prompt(()),
                ),
                StepAndAnswer(
                    dataset.SymbolSetStep(state_subset="dataset_0"),
                    null_patch(),
                ),
                StepAndAnswer(
                    dataset.SoxEffectsStep(state_subset="dataset_0"),
                    patch_menu_prompt([]),
                ),
                StepAndAnswer(
                    dataset.DatasetNameStep(state_subset="dataset_0"),
                    patch_input("dataset0"),
                ),
                StepAndAnswer(
                    basic.MoreDatasetsStep(),
                    patch_menu_prompt(1),  # 1 is yes
                    children_answers=more_dataset_children_answers,
                ),
            ]
            _tour, _ = self.monkey_run_tour(
                "Tour with datafile in the festival format",
                steps_and_answers,
            )
            with open(
                tmpdir / "out/project/dataset0-filelist.psv", encoding="utf8"
            ) as f:
                output_filelist = [line.rstrip() for line in f]
            expected_filelist1 = [
                "basename|language|speaker|characters|phones",
                "f1|eng|speaker_1|foo foo|fu fu",
                "f2|eng|speaker_1|bar bar|bɑɹ bɑɹ",
                "f3|eng|speaker_1|baz baz|bæz bæz",
            ]
            self.assertListEqual(output_filelist, expected_filelist1)

            with open(
                tmpdir / "out/project/dataset1-filelist.psv", encoding="utf8"
            ) as f:
                output_filelist = [line.rstrip() for line in f]
            expected_filelist2 = [
                "basename|language|speaker|characters|phones",
                "f4|und|default_speaker|foo bar|foo bar",
                "f5|und|default_speaker|bar baz|bar baz",
                "f6|und|default_speaker|baz foo|baz foo",
            ]
            self.assertListEqual(output_filelist, expected_filelist2)

            with open(
                tmpdir / "out/project/config/everyvoice-text-to-spec.yaml",
                encoding="utf8",
            ) as f:
                text_to_spec_config = "\n".join(f)
            self.assertIn("multilingual: true", text_to_spec_config)
            self.assertIn("multispeaker: true", text_to_spec_config)

    def test_multilingual_multispeaker_false_config(self):
        """
        Test matched multi-monospeaker and multi-monolingual datasets.

        Makes sure that multilingual and multispeaker parameters of config are set to false when two monolingual and monospeaker datasets are provided with the specified languages and speakers which are the same.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with open(tmpdir / "filelist1.psv", "w", encoding="utf8") as f:
                f.write(
                    "\n".join(
                        [
                            "basename|language|text",
                            "f1|und|foo foo",
                            "f2|und|bar bar",
                            "f3|und|baz baz",
                        ]
                    )
                )
            with open(tmpdir / "filelist2.txt", "w", encoding="utf8") as f:
                f.write(
                    "\n".join(
                        [
                            '( f4 " foo bar " )',
                            '( f5 " bar baz " )',
                            '( f6 " baz foo " )',
                        ]
                    )
                )
            for basename in ("f1", "f2", "f3", "f4", "f5", "f6"):
                with open(tmpdir / (basename + ".wav"), "wb"):
                    pass
            more_dataset_children_answers = [
                RecursiveAnswers(
                    patch_questionary(tmpdir / "filelist2.txt")
                ),  # filelist step
                RecursiveAnswers(
                    patch_menu_prompt(1),  # yes to permission
                    children_answers=[
                        RecursiveAnswers(patch_menu_prompt(3)),  # festival format
                        RecursiveAnswers(patch_menu_prompt(0)),  # characters
                        RecursiveAnswers(
                            patch_menu_prompt(())
                        ),  # no text preprocessing
                        RecursiveAnswers(
                            null_patch(),  # skip the question about a speaker column in the data since its festival
                            children_answers=[
                                RecursiveAnswers(
                                    patch_menu_prompt(
                                        1
                                    ),  # want to specify the speaker ID
                                    children_answers=[
                                        RecursiveAnswers(
                                            patch_input("speaker_0")
                                        )  # new speaker ID
                                    ],
                                ),
                            ],
                        ),
                        RecursiveAnswers(
                            null_patch(),  # skip the question about a language column in the data since its festival
                            children_answers=[
                                RecursiveAnswers(
                                    patch_menu_prompt(0),  # "und" lang selection
                                )
                            ],
                        ),
                        RecursiveAnswers(patch_menu_prompt(0)),  # Keep g2p settings
                        RecursiveAnswers(patch_questionary(tmpdir)),  # wav directory
                        RecursiveAnswers(null_patch()),  # ValidateWavsStep
                        RecursiveAnswers(null_patch()),  # SymbolSetStep
                        RecursiveAnswers(patch_menu_prompt([])),  # no Sox
                        RecursiveAnswers(patch_input("dataset1")),  # Dataset name
                    ],
                ),
                RecursiveAnswers(
                    patch_menu_prompt(0),
                    children_answers=[RecursiveAnswers(patch_menu_prompt(0))],
                ),
            ]
            steps_and_answers = [
                StepAndAnswer(basic.NameStep(), patch_input("project")),
                StepAndAnswer(basic.ContactNameStep(), patch_input("Test Name")),
                StepAndAnswer(
                    basic.ContactEmailStep(), patch_input("info@everyvoice.ca")
                ),
                StepAndAnswer(
                    basic.OutputPathStep(), patch_questionary(tmpdir / "out")
                ),
                # First dataset
                StepAndAnswer(
                    dataset.WavsDirStep(state_subset="dataset_0"),
                    patch_questionary(tmpdir),
                ),
                StepAndAnswer(
                    dataset.FilelistStep(state_subset="dataset_0"),
                    patch_questionary(tmpdir / "filelist1.psv"),
                ),
                StepAndAnswer(
                    dataset.FilelistFormatStep(state_subset="dataset_0"),
                    patch_menu_prompt(0),  # psv
                ),
                StepAndAnswer(
                    dataset.FilelistTextRepresentationStep(state_subset="dataset_0"),
                    patch_menu_prompt(0),  # characters
                ),
                StepAndAnswer(
                    dataset.HasSpeakerStep(state_subset="dataset_0"),
                    patch_menu_prompt(0),  # 0 is no
                    children_answers=[
                        RecursiveAnswers(patch_menu_prompt(0)),  # 0 is no
                    ],
                ),
                StepAndAnswer(
                    dataset.HasLanguageStep(state_subset="dataset_0"),
                    patch_menu_prompt(1),  # 1 is yes
                    children_answers=[RecursiveAnswers(Say(1))],
                ),
                StepAndAnswer(
                    dataset.CustomG2PStep(state_subset="dataset_0"),
                    patch_menu_prompt(0),  # 0 is Keep the current g2p settings
                ),
                StepAndAnswer(
                    dataset.TextProcessingStep(state_subset="dataset_0"),
                    patch_menu_prompt(()),
                ),
                StepAndAnswer(
                    dataset.SymbolSetStep(state_subset="dataset_0"),
                    null_patch(),
                ),
                StepAndAnswer(
                    dataset.SoxEffectsStep(state_subset="dataset_0"),
                    patch_menu_prompt([]),
                ),
                StepAndAnswer(
                    dataset.DatasetNameStep(state_subset="dataset_0"),
                    patch_input("dataset0"),
                ),
                StepAndAnswer(
                    basic.MoreDatasetsStep(),
                    patch_menu_prompt(1),  # 1 is yes
                    children_answers=more_dataset_children_answers,
                ),
            ]
            _tour, _ = self.monkey_run_tour(
                "Tour with datafile in the festival format",
                steps_and_answers,
            )
            with open(
                tmpdir / "out/project/dataset0-filelist.psv", encoding="utf8"
            ) as f:
                output_filelist = [line.rstrip() for line in f]
            expected_filelist1 = [
                "basename|language|speaker|characters|phones",
                "f1|und|speaker_0|foo foo|foo foo",
                "f2|und|speaker_0|bar bar|bar bar",
                "f3|und|speaker_0|baz baz|baz baz",
            ]
            self.assertListEqual(output_filelist, expected_filelist1)

            with open(
                tmpdir / "out/project/dataset1-filelist.psv", encoding="utf8"
            ) as f:
                output_filelist = [line.rstrip() for line in f]
            expected_filelist2 = [
                "basename|language|speaker|characters|phones",
                "f4|und|speaker_0|foo bar|foo bar",
                "f5|und|speaker_0|bar baz|bar baz",
                "f6|und|speaker_0|baz foo|baz foo",
            ]
            self.assertListEqual(output_filelist, expected_filelist2)

            with open(
                tmpdir / "out/project/config/everyvoice-text-to-spec.yaml",
                encoding="utf8",
            ) as f:
                text_to_spec_config = "\n".join(f)
            self.assertIn("multilingual: false", text_to_spec_config)
            self.assertIn("multispeaker: false", text_to_spec_config)

    trivial_tour_results = {
        SN.name_step.value: "project_name",
        SN.contact_name_step.value: "Jane Doe",
        SN.contact_email_step.value: "email@mail.com",
    }

    def test_control_c_go_back(self):
        # Ctrl-C plus option 0 goes back
        tour = make_trivial_tour()
        with patch_input(
            [
                "bad_name",
                KeyboardInterrupt(),
                "project_name",
                "bad user name",
                KeyboardInterrupt(),
                "Jane Doe",
                "email@mail.com",
            ],
            multi=True,
        ):
            with patch_menu_prompt(0):  # say 0==go back each time
                tour.run()
        self.assertEqual(tour.state, self.trivial_tour_results)

    def test_control_c_continue(self):
        # Ctrl-C plus option 1 continues
        tour = make_trivial_tour()
        with patch_input(
            ["project_name", KeyboardInterrupt(), "Jane Doe", "email@mail.com"],
            multi=True,
        ):
            # Ctrl-C once, then hit 1 to continue
            with patch_menu_prompt([KeyboardInterrupt(), 1], multi=True):
                tour.run()
        self.assertEqual(tour.state, self.trivial_tour_results)

    def test_control_c_display_tree(self):
        # Ctrl-C plus option 2 displays the current tree
        tour = make_trivial_tour()
        with patch_input(
            ["project_name", "Jane Doe", KeyboardInterrupt(), "email@mail.com"],
            multi=True,
        ):
            with patch_menu_prompt(2) as output:
                tour.run()
            self.assertRegex(output.getvalue(), r"Contact Name: *Jane Doe")
        self.assertEqual(tour.state, self.trivial_tour_results)

    progress_template = dedent(
        """\
        - - EveryVoice Wizard
          - {version}
        - - Root
          - null
        - - Name Step
          - project_name
        - - Contact Name Step
          - Jane Doe
        """
    )

    def test_control_c_save_progress(self):
        # Ctrl-C plus option 3 saves progress to file
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            tour = make_trivial_tour()
            progress_file = tmpdir / "saved-progress"
            with (
                patch_input(
                    [
                        "project_name",
                        "Jane Doe",
                        KeyboardInterrupt(),
                        "email@mail.com",
                    ],
                    multi=True,
                ),
                patch_questionary(str(progress_file), ask_ok=True),
            ):
                with patch_menu_prompt(3):
                    tour.run()
            self.assertTrue(progress_file.exists())
            with open(progress_file, encoding="utf8") as f:
                progress_contents = f.read()
                # print(progress_contents)
                self.assertEqual(
                    progress_contents, self.progress_template.format(version=VERSION)
                )

    def test_resume_from(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            progress_file = tmpdir / "saved-progress"
            with open(progress_file, "w") as f:
                f.write(self.progress_template.format(version=VERSION))
            # resume works
            tour = make_trivial_tour()
            with patch_input("email@mail.com"), capture_stdout() as out:
                tour.run(resume_from=progress_file)
            self.assertIn("Applying saved response", out.getvalue())
            self.assertEqual(tour.state, self.trivial_tour_results)

    def test_resume_from_the_future(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            # resume from a future version works with a warning
            changed_version = tmpdir / "changed-version"
            with open(changed_version, "w", encoding="utf8") as f:
                v = Version(VERSION)
                f.write(
                    self.progress_template.format(
                        version=f"{v.major + 1}.{v.minor}.{v.micro}"
                    )
                )

            tour = make_trivial_tour()
            with patch_input("email@mail.com"), capture_stdout() as out:
                tour.run(resume_from=changed_version)
            self.assertRegex(out.getvalue(), r"(?s)Proceeding.*anyway")
            self.assertRegex(out.getvalue(), r"(?s)consider.*updating.*your.*software")
            self.assertIn("Applying saved response", out.getvalue())
            self.assertEqual(tour.state, self.trivial_tour_results)

    def test_resume_from_near_past(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            # resume from a changed version works with a warning
            changed_version = tmpdir / "changed-version"
            with open(changed_version, "w", encoding="utf8") as f:
                recent_past = VERSION + ".dev0" if "a" in VERSION else VERSION + ".pre0"
                f.write(self.progress_template.format(version=recent_past))

            tour = make_trivial_tour()
            with patch_input("email@mail.com"), capture_stdout() as out:
                tour.run(resume_from=changed_version)
            self.assertRegex(out.getvalue(), r"(?s)expected.*to.*be.*compatible")
            self.assertRegex(out.getvalue(), r"(?s)Proceeding.*anyway")
            self.assertIn("Applying saved response", out.getvalue())
            self.assertEqual(tour.state, self.trivial_tour_results)

    def test_resume_from_far_past(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            # resume from a potentially incompatible older version
            changed_version = tmpdir / "changed-version"
            with open(changed_version, "w", encoding="utf8") as f:
                f.write(self.progress_template.format(version="0.1.2"))
            tour = make_trivial_tour()
            with patch_input("email@mail.com"), capture_stdout() as out:
                tour.run(resume_from=changed_version)
            self.assertRegex(out.getvalue(), r"(?s)not.*fully.*compatible")
            self.assertRegex(out.getvalue(), r"(?s)Proceeding.*anyway")
            self.assertIn("Applying saved response", out.getvalue())
            self.assertEqual(tour.state, self.trivial_tour_results)

    def test_resume_with_invalid_progress_files(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)

            # This one has an invalid response but lets the user recover
            invalid_response = tmpdir / "invalid-response"
            with open(invalid_response, "w", encoding="utf8") as f:
                f.write(self.progress_template.format(version=VERSION))
                f.write("- - Contact Email Step\n  - invalid email\n")
            tour = make_trivial_tour()
            with patch_input("email@mail.com"), capture_stdout() as out:
                tour.run(resume_from=invalid_response)
            self.assertIn("Error: saved response 'invalid email'", out.getvalue())
            self.assertEqual(tour.state, self.trivial_tour_results)

            # From here on, it's lots of ways to fail, which always causes a SystemExit

            bad_progress_file = tmpdir / "bad-progress"
            with open(bad_progress_file, "w", encoding="utf8") as f:
                f.write("not a list of questions and answers")
            with self.assertRaises(SystemExit), capture_stdout():
                tour.run(resume_from=bad_progress_file)

            bad_progress_file2 = tmpdir / "bad-progress2"
            with open(bad_progress_file2, "w", encoding="utf8") as f:
                f.write("- - question 1\n  - answer 1\n  - extra garbage\n")
            with self.assertRaises(SystemExit), capture_stdout():
                tour.run(resume_from=bad_progress_file2)

            progress_lines = self.progress_template.format(version=VERSION).splitlines(
                keepends=True
            )

            truncated_progress_file = tmpdir / "truncated-progress"
            with open(truncated_progress_file, "w", encoding="utf8") as f:
                f.write("".join(progress_lines[:-1]))
            with self.assertRaises(SystemExit), capture_stdout():
                tour.run(resume_from=truncated_progress_file)

            missing_first_line = tmpdir / "missing-first-line"
            with open(missing_first_line, "w", encoding="utf8") as f:
                f.write("".join(progress_lines[1:]))
            with self.assertRaises(SystemExit), capture_stdout():
                tour.run(resume_from=missing_first_line)

            with self.assertRaises(SystemExit), capture_stdout():
                tour.run(resume_from=Path(os.devnull))

            questions_out_of_order = tmpdir / "questions-out-of-order"
            with open(questions_out_of_order, "w", encoding="utf8") as f:
                f.write("".join(progress_lines[:-4]))
                f.write("".join(progress_lines[-2:]))
                f.write("".join(progress_lines[-4:-2]))
            with self.assertRaises(SystemExit), capture_stdout() as out:
                tour.run(resume_from=questions_out_of_order)
            self.assertIn("out of sync", out.getvalue())

            extra_question_not_in_tour = tmpdir / "extra-question"
            with open(extra_question_not_in_tour, "w", encoding="utf8") as f:
                f.write("".join(progress_lines))
                f.write("- - Contact Email Step\n  - email@mail.com\n")
                f.write("- - Not a real Step\n  - bogus answer\n")
            tour = make_trivial_tour()
            with self.assertRaises(SystemExit), capture_stdout() as out:
                tour.run(resume_from=extra_question_not_in_tour)
            self.assertIn("saved responses left", out.getvalue())

            wrong_software_name = tmpdir / "wrong-software-name"
            with open(wrong_software_name, "w", encoding="utf8") as f:
                f.write("- - Wrong Software\n")
                f.write("".join(progress_lines[1:]))
            with self.assertRaises(SystemExit), capture_stdout() as out:
                tour.run(resume_from=wrong_software_name)
            self.assertRegex(out.getvalue(), r"(?s)it.*is.*for.*software")

    def test_control_c_exit(self):
        # Ctrl-C plus option 4 (Exit) exits
        tour = make_trivial_tour()
        with patch_input(KeyboardInterrupt()):
            with patch_menu_prompt(4):  # 4 is "Exit" in keyboard interrupt handling
                with self.assertRaises(SystemExit):
                    tour.run()

        # Three Ctrl-C also exits
        tour = make_trivial_tour()
        with patch_input(KeyboardInterrupt()), patch_menu_prompt(KeyboardInterrupt()):
            with self.assertRaises(SystemExit):
                tour.run()

    def test_trace(self):
        tour = make_trivial_tour(trace=True)
        with patch_input(["project_name", "user name", "email@mail.com"], multi=True):
            with capture_stdout() as out:
                tour.run()
        for step in tour.steps:
            # When not the current step:
            self.assertIn(step.name.replace(" Step", "") + "  ", out.getvalue())
            # When it is the current step:
            self.assertRegex(out.getvalue(), step.name.replace(" Step", "") + " *←")
            # When previously filled:
            if step != tour.steps[-1]:
                self.assertIn(step.name.replace(" Step", "") + ": ", out.getvalue())

    def test_debug_state(self):
        tour = make_trivial_tour(debug_state=True)
        responses = ["project_name", "user name", "email@mail.com"]
        with patch_input(responses, multi=True):
            with capture_stdout() as out:
                tour.run()
        for step, response in zip(tour.steps, responses[:-1]):
            # When not the current step:
            self.assertRegex(out.getvalue(), f"'{step.name}'.*: .*'{response}'")
