#!/usr/bin/env python

import string
import tempfile
from pathlib import Path
from types import MethodType
from unittest import TestCase, main

from anytree import RenderTree

from everyvoice.config.text_config import Symbols
from everyvoice.wizard import Step, StepNames
from everyvoice.wizard.basic import ConfigFormatStep


class WizardTest(TestCase):
    """Basic test for the config wizard"""

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
        config_step = ConfigFormatStep(name="Config Step")
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
