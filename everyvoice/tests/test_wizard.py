#!/usr/bin/env python

import logging
import string
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from unittest import TestCase, main

from anytree import RenderTree

import everyvoice.wizard.basic as basic
from everyvoice.config.text_config import Symbols
from everyvoice.wizard import Step, StepNames, Tour


@contextmanager
def patch_logger(module):
    save_logger = module.logger
    try:
        module.logger = logging.getLogger("UnitTesting")
        yield module.logger
    finally:
        module.logger = save_logger


def canned_response(response):
    """Return a function that returns response regardless of its input."""

    def helper(*args, **kwargs):
        return response

    return helper


@contextmanager
def patch_response(response):
    # Note the white-box ismplementation detail: since basic.py does
    #    from everyvoice.wizard.prompts import get_response_from_menu_prompt
    # we have to monkey patch the imported copy in basic.py rather that the
    # original in prompts.py
    save_response = basic.get_response_from_menu_prompt
    try:
        basic.get_response_from_menu_prompt = canned_response(response)
        yield
    finally:
        basic.get_response_from_menu_prompt = save_response


@contextmanager
def patch_prompt(step, response):
    save_prompt = step.prompt
    try:
        step.prompt = canned_response(response)
        yield
    finally:
        step.prompt = save_prompt


class WizardTest(TestCase):
    """Basic test for the new dataset wizard"""

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
            config_step.state[StepNames.output_step.value] = tmpdirname
            config_step.state[StepNames.name_step.value] = config_step.name
            config_step.state["dataset_test"] = {}
            config_step.state["dataset_test"][
                StepNames.symbol_set_step.value
            ] = Symbols(symbol_set=string.ascii_letters)
            config_step.state["dataset_test"][StepNames.wavs_dir_step.value] = (
                Path(tmpdirname) / "test"
            )
            config_step.state["dataset_test"][
                StepNames.dataset_name_step.value
            ] = "test"
            config_step.state["dataset_test"]["filelist_data"] = [
                {"basename": "0001", "text": "hello"},
                {"basename": "0002", "text": "hello", None: "test"},
            ]
            config_step.state["dataset_test"]["sox_effects"] = []
            config_step.effect()
            self.assertTrue(
                (Path(tmpdirname) / config_step.name / "logs_and_checkpoints").exists()
            )

    def test_name_step(self):
        """Exercise provide a valid dataset name."""
        step = basic.NameStep("")
        with patch_logger(basic) as logger:
            with self.assertLogs(logger) as logs:
                with patch_prompt(step, "myname"):
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
        with patch_logger(basic) as logger:
            with self.assertLogs(logger) as logs:
                self.assertFalse(step.validate("foo/bar"))
        self.assertIn("'foo/bar' is not valid", logs.output[0])

    def test_more_data_step(self):
        """Exercise giving an invalid response and a yes response to more data."""
        tour = Tour(
            "testing", [basic.MoreDatasetsStep(name=StepNames.more_datasets_step.value)]
        )
        step = tour.steps[0]
        self.assertFalse(step.validate("foo"))
        self.assertTrue(step.validate("yes"))
        self.assertEqual(len(step.children), 0)
        with patch_response("no"):
            step.run()
        self.assertEqual(len(step.children), 1)
        self.assertIsInstance(step.children[0], basic.ConfigFormatStep)

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


if __name__ == "__main__":
    main()
