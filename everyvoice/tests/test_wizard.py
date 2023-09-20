#!/usr/bin/env python

import builtins
import os
import string
import tempfile
from enum import Enum
from pathlib import Path
from types import MethodType
from typing import Sequence
from unittest import TestCase, main

from anytree import RenderTree

from everyvoice.config import preprocessing_config
from everyvoice.config.text_config import Symbols
from everyvoice.tests.stubs import (
    QUIET,
    QuestionaryStub,
    Say,
    capture_stdout,
    monkeypatch,
    patch_logger,
    patch_menu_prompt,
)
from everyvoice.wizard import Step
from everyvoice.wizard import StepNames as SN
from everyvoice.wizard import Tour, basic, dataset, prompts, validators


class WizardTest(TestCase):
    """Basic test for the new dataset wizard"""

    data_dir = Path(__file__).parent / "data"

    def setUp(self) -> None:
        pass

    def test_implementation_missing(self):
        nothing_step = Step(name="Dummy Step")
        no_validate_step = Step(name="Dummy Step", prompt_method=lambda: "test")
        no_prompt_step = Step(name="Dummy Step", validate_method=lambda: True)
        for step in [nothing_step, no_validate_step, no_prompt_step]:
            with self.assertRaises(NotImplementedError):
                step.run()

    def test_config_format_effect(self):
        """This is testing is that a null key can be passed without throwing an error,
        as reported by Marc Tessier. There are no assertions, it is just testing that
        no exceptions get raised.
        """
        config_step = basic.ConfigFormatStep(name="Config Step")
        self.assertTrue(config_step.validate("yaml"))
        self.assertTrue(config_step.validate("json"))
        with tempfile.TemporaryDirectory() as tmpdirname:
            config_step.state = {}
            config_step.state[SN.output_step.value] = tmpdirname
            config_step.state[SN.name_step.value] = config_step.name
            config_step.state["dataset_test"] = {}
            config_step.state["dataset_test"][SN.symbol_set_step.value] = Symbols(
                symbol_set=string.ascii_letters
            )
            config_step.state["dataset_test"][SN.wavs_dir_step.value] = (
                Path(tmpdirname) / "test"
            )
            config_step.state["dataset_test"][SN.dataset_name_step.value] = "test"
            config_step.state["dataset_test"]["filelist_data"] = [
                {"basename": "0001", "text": "hello"},
                {"basename": "0002", "text": "hello", None: "test"},
            ]
            config_step.state["dataset_test"]["sox_effects"] = []
            with patch_logger(preprocessing_config, QUIET):
                with capture_stdout() as stdout:
                    config_step.effect()
            self.assertIn("Congratulations", stdout.getvalue())
            self.assertTrue(
                (Path(tmpdirname) / config_step.name / "logs_and_checkpoints").exists()
            )

    def test_access_response(self):
        root_step = Step(
            name="Dummy Step",
            prompt_method=lambda: "foo",
            validate_method=lambda x: True,
        )

        def validate(self, x):
            """Because the root step always returns 'foo', we only validate the second step if the prompt returns 'foobar'"""
            if self.parent.response + x == "foobar":
                return True

        second_step = Step(
            name="Dummy Step 2", prompt_method=lambda: "bar", parent=root_step
        )
        second_step.validate = MethodType(validate, second_step)
        for i, leaf in enumerate(RenderTree(root_step)):
            if i != 0:
                self.assertEqual(second_step.parent.response, "foo")
                self.assertTrue(leaf[2].validate("bar"))
                self.assertFalse(leaf[2].validate("foo"))
            leaf[2].run()

    def test_main_tour(self):
        from everyvoice.wizard.main_tour import WIZARD_TOUR

        tour = WIZARD_TOUR
        self.assertGreater(len(tour.steps), 6)
        # TODO try to figure out how to actually run the tour in unit testing or
        # at least add more interesting assertions that just the fact that it's
        # got several steps.

    def test_name_step(self):
        """Exercise provide a valid dataset name."""
        step = basic.NameStep("")
        with patch_logger(basic) as logger, self.assertLogs(logger) as logs:
            with monkeypatch(builtins, "input", Say("myname")):
                step.run()
        self.assertEqual(step.response, "myname")
        self.assertIn("named 'myname'", logs.output[0])
        self.assertTrue(step.completed)

    def test_bad_name_step(self):
        """Exercise provide an invalid dataset name."""
        step = basic.NameStep("")
        # For unit testing, we cannot call step.run() if we patch a response
        # that will fail validation, because that would cause infinite
        # recursion.
        with patch_logger(basic) as logger, self.assertLogs(logger) as logs:
            self.assertFalse(step.validate("foo/bar"))
            self.assertFalse(step.validate(""))
        self.assertIn("'foo/bar' is not valid", logs.output[0])
        self.assertIn("you have to put something", logs.output[1])

        step = basic.NameStep("")
        with patch_logger(basic) as logger, self.assertLogs(logger) as logs:
            with monkeypatch(builtins, "input", Say(("bad/name", "good-name"), True)):
                step.run()
        self.assertIn("'bad/name' is not valid", logs.output[0])
        self.assertEqual(step.response, "good-name")

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
        with patch_logger(basic, level=QUIET) as logger:
            with monkeypatch(builtins, "input", Say("myname")):
                step.run()

        step = tour.steps[1]
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, "exits-as-file")
            # Bad case 1: output dir exists and is a file
            with open(file_path, "w", encoding="utf8") as f:
                f.write("blah")
            with patch_logger(basic) as logger, self.assertLogs(logger):
                self.assertFalse(step.validate(file_path))

            # Bad case 2: file called the same as the dataset exists in the output dir
            dataset_file = os.path.join(tmpdirname, "myname")
            with open(dataset_file, "w", encoding="utf8") as f:
                f.write("blah")
            with patch_logger(basic) as logger, self.assertLogs(logger):
                self.assertFalse(step.validate(tmpdirname))
            os.unlink(dataset_file)

            # Good case
            with patch_logger(basic) as logger, self.assertLogs(logger) as logs:
                with monkeypatch(step, "prompt", Say(tmpdirname)):
                    step.run()
            self.assertIn("will put your files", logs.output[0])
            output_dir = Path(tmpdirname) / "myname"
            self.assertTrue(output_dir.exists())
            self.assertTrue(output_dir.is_dir())

    def test_more_data_step(self):
        """Exercise giving an invalid response and a yes response to more data."""
        tour = Tour("testing", [basic.MoreDatasetsStep()])
        step = tour.steps[0]
        self.assertFalse(step.validate("foo"))
        self.assertTrue(step.validate("yes"))
        self.assertEqual(len(step.children), 0)
        with patch_menu_prompt(1):  # answer 1 is "no"
            step.run()
        self.assertEqual(len(step.children), 1)
        self.assertIsInstance(step.children[0], basic.ConfigFormatStep)

        with patch_menu_prompt(0):  # answer 0 is "yes"
            step.run()
        self.assertGreater(len(step.children), 5)

    def test_dataset_name(self):
        step = dataset.DatasetNameStep()
        with monkeypatch(builtins, "input", Say(("", "bad/name", "good-name"), True)):
            with patch_logger(dataset) as logger, self.assertLogs(logger) as logs:
                step.run()
        self.assertIn("you have to put something here", logs.output[0])
        self.assertIn("is not valid", logs.output[1])
        self.assertIn("finished the configuration", logs.output[2])
        self.assertTrue(step.completed)

    def test_wavs_dir(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            no_wavs_dir = os.path.join(tmpdirname, "no-wavs-here")
            os.mkdir(no_wavs_dir)

            has_wavs_dir = os.path.join(tmpdirname, "there-are-wavs-here")
            os.mkdir(has_wavs_dir)
            with open(os.path.join(has_wavs_dir, "foo.wav"), "wb") as f:
                f.write(b"A fantastic sounding clip! (or not...)")

            step = dataset.WavsDirStep("")
            with monkeypatch(
                dataset,
                "questionary",
                QuestionaryStub(("not-a-path", no_wavs_dir, has_wavs_dir)),
            ):
                with patch_logger(validators, QUIET):
                    step.run()
            self.assertTrue(step.completed)
            self.assertEqual(step.response, has_wavs_dir)

    def test_sample_rate_config(self):
        step = dataset.SampleRateConfigStep("")
        with monkeypatch(
            dataset,
            "questionary",
            QuestionaryStub(
                (
                    "not an int",  # obvious not valid
                    "",  # ditto
                    3.1415,  # floats are also not allowed
                    50,  # this is below the minimum 100 allowed
                    512,  # yay, a good response!
                    1024,  # won't get used becasue 512 is good.
                )
            ),
        ):
            with patch_logger(dataset) as logger, self.assertLogs(logger) as logs:
                step.run()
        for i in range(4):
            self.assertIn("not a valid sample rate", logs.output[i])
        self.assertTrue(step.completed)
        self.assertEqual(step.response, 512)

    def test_dataset_subtour(self):
        def find_step(name: Enum, steps: Sequence[Step]):
            for s in steps:
                if s.name == name.value:
                    return s
            raise IndexError(f"Step {name} not found.")  # pragma: no cover

        tour = Tour("unit testing", steps=dataset.return_dataset_steps())
        filelist = str(self.data_dir / "unit-test-case1.psv")

        filelist_step = find_step(SN.filelist_step, tour.steps)
        monkey = monkeypatch(filelist_step, "prompt", Say(filelist))
        with monkey:
            filelist_step.run()
        format_step = find_step(SN.filelist_format_step, tour.steps)
        with patch_menu_prompt(0):  # 0 is "psv"
            format_step.run()
        self.assertIsInstance(format_step.children[0], dataset.HeaderStep)
        self.assertEqual(format_step.children[0].name, SN.basename_header_step.value)
        self.assertIsInstance(format_step.children[1], dataset.HeaderStep)
        self.assertEqual(format_step.children[1].name, SN.text_header_step.value)

        step = format_step.children[0]
        with patch_menu_prompt(1):  # 1 is second column
            step.run()
        self.assertEqual(step.response, 1)
        self.assertEqual(step.state["filelist_headers"][1], "basename")

        step = tour.steps[2].children[1]
        with patch_menu_prompt(1):  # 1 is second remaining column, i.e., third column
            step.run()
        # print(step.state["filelist_headers"])
        self.assertEqual(step.state["filelist_headers"][2], "text")

        speaker_step = find_step(SN.data_has_speaker_value_step, tour.steps)
        children_before = len(speaker_step.children)
        with patch_menu_prompt(1):  # 1 is "no"
            speaker_step.run()
        self.assertEqual(len(speaker_step.children), children_before)

        language_step = find_step(SN.data_has_language_value_step, tour.steps)
        children_before = len(language_step.children)
        with patch_menu_prompt(1):  # 1 is "no"
            language_step.run()
        self.assertEqual(len(language_step.children), children_before + 1)
        self.assertIsInstance(language_step.children[0], dataset.SelectLanguageStep)

        select_lang_step = language_step.children[0]
        with patch_logger(dataset, QUIET):
            with patch_menu_prompt(15):  # some arbitrary language from the list
                select_lang_step.run()
        # print(select_lang_step.state)
        self.assertEqual(
            select_lang_step.state["filelist_headers"],
            ["unknown_0", "basename", "text", "unknown_3"],
        )

        text_processing_step = find_step(SN.text_processing_step, tour.steps)
        # 0 is lowercase, 1 is NFC Normalization, select both
        with monkeypatch(dataset, "tqdm", lambda seq, desc: seq):
            with patch_menu_prompt([0, 1]):
                text_processing_step.run()
        # print(text_processing_step.state)
        self.assertEqual(
            text_processing_step.state["filelist_data"][2]["text"],
            "cased \t nfd: éàê nfc: éàê",  # the "nfd: éàê" bit here is now NFC
        )

        sox_effects_step = find_step(SN.sox_effects_step, tour.steps)
        # 0 is resample to 22050 kHz, 2 is remove silence at start
        with patch_menu_prompt([0, 2]):
            sox_effects_step.run()
        # print(sox_effects_step.state["sox_effects"])
        self.assertEqual(
            sox_effects_step.state["sox_effects"],
            [["channel", "1"], ["rate", "22050"], ["silence", "1", "0.1", "1.0%"]],
        )

        symbol_set_step = find_step(SN.symbol_set_step, tour.steps)
        self.assertEqual(len(symbol_set_step.state["filelist_data"]), 4)
        with patch_menu_prompt([(0, 1, 2, 3), (11), ()], multi=True):
            symbol_set_step.run()
        self.assertEqual(symbol_set_step.state["banned_symbols"], "z")
        self.assertEqual(symbol_set_step.response.punctuation, ["\t", " ", "-", ":"])
        self.assertEqual(
            symbol_set_step.response.symbol_set,
            ["a", "c", "d", "e", "f", "n", "o", "r", "s", "t", "x", "à", "é", "ê"],
        )
        self.assertEqual(len(symbol_set_step.state["filelist_data"]), 3)

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
        with patch_logger(dataset) as logger, self.assertLogs(logger) as logs:
            with patch_menu_prompt((1, 2, 3, 0), multi=True):
                format_step.run()
        self.assertIn("does not look like a tsv", logs.output[0])
        self.assertIn("does not look like a csv", logs.output[1])
        self.assertIn("festival", logs.output[2])
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
        with patch_logger(dataset) as logger, self.assertLogs(logger) as logs:
            with patch_menu_prompt((0, 1, 2, 3), multi=True):
                format_step.run()
        self.assertIn("does not look like a psv", logs.output[0])
        self.assertIn("does not look like a tsv", logs.output[1])
        self.assertIn("does not look like a csv", logs.output[2])
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
        with tempfile.TemporaryDirectory() as tmpdirname, patch_logger(
            validators, QUIET
        ):
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

    def monkey_run_tour(self, name, steps):
        tour = Tour(name, steps=[step for (step, *_) in steps])
        self.assertEqual(tour.state, {})  # fail on accidentally shared initializer
        for (step, answer, *_) in steps:
            if isinstance(answer, Say):
                monkey = monkeypatch(step, "prompt", answer)
            else:
                monkey = answer
            # print(step.name)
            with monkey:
                step.run()
        return tour

    def test_monkey_tour_1(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tour = self.monkey_run_tour(
                "monkey tour 1",
                [
                    (basic.NameStep(), Say("my-dataset-name")),
                    (basic.OutputPathStep(), Say(tmpdirname)),
                ],
            )
        self.assertEqual(tour.state[SN.name_step.value], "my-dataset-name")
        self.assertEqual(tour.state[SN.output_step.value], tmpdirname)

    def test_monkey_tour_2(self):
        data_dir = Path(__file__).parent / "data"
        tour = self.monkey_run_tour(
            "monkey tour 2",
            [
                (dataset.WavsDirStep(), Say(data_dir)),
                (
                    dataset.FilelistStep(),
                    Say(str(data_dir / "metadata.csv")),
                ),
                (dataset.FilelistFormatStep(), Say("psv")),
                (dataset.HasSpeakerStep(), Say("yes")),
                (dataset.HasLanguageStep(), Say("yes")),
                (dataset.SelectLanguageStep(), Say("eng")),
                (dataset.TextProcessingStep(), Say([0, 1])),
                (
                    dataset.SymbolSetStep(),
                    patch_menu_prompt([(0, 1, 2, 3, 4), (), ()], multi=True),
                ),
                (dataset.SoxEffectsStep(), Say([0])),
                (dataset.DatasetNameStep(), Say("my-monkey-dataset")),
            ],
        )

        # print(tour.state)
        self.assertEqual(len(tour.state["filelist_data"]), 6)


if __name__ == "__main__":
    main()
