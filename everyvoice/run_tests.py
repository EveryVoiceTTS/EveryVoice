#!/usr/bin/env python

""" Organize tests into Test Suites
"""

import os
import sys
from unittest import TestLoader, TestSuite, TextTestRunner

from loguru import logger

from everyvoice.tests.test_cli import CLITest
from everyvoice.tests.test_configs import ConfigTest, LoadConfigTest
from everyvoice.tests.test_dataloader import DataLoaderTest
from everyvoice.tests.test_model import ModelTest
from everyvoice.tests.test_preprocessing import (
    PreprocessingHierarchyTest,
    PreprocessingTest,
)
from everyvoice.tests.test_text import TextTest
from everyvoice.tests.test_wizard import WavFileDirectoryRelativePathTest, WizardTest

# Unit tests


LOADER = TestLoader()

CONFIG_TESTS = [
    LOADER.loadTestsFromTestCase(test) for test in [ConfigTest, LoadConfigTest]
]

DATALOADER_TESTS = [LOADER.loadTestsFromTestCase(test) for test in [DataLoaderTest]]

TEXT_TESTS = [LOADER.loadTestsFromTestCase(test) for test in [TextTest]]

PREPROCESSING_TESTS = [
    LOADER.loadTestsFromTestCase(test)
    for test in [PreprocessingTest, PreprocessingHierarchyTest]
]

MODEL_TESTS = [LOADER.loadTestsFromTestCase(test) for test in [ModelTest]]

CLI_TESTS = [
    LOADER.loadTestsFromTestCase(test)
    for test in [WizardTest, CLITest, WavFileDirectoryRelativePathTest]
]

DEV_TESTS = (
    CONFIG_TESTS
    + DATALOADER_TESTS
    + TEXT_TESTS
    + PREPROCESSING_TESTS
    + MODEL_TESTS
    + CLI_TESTS
)


def run_tests(suite):
    """Decide which Test Suite to run"""
    if suite == "all":
        suite = LOADER.discover(os.path.dirname(__file__))
    elif suite == "configs":
        suite = TestSuite(CONFIG_TESTS)
    elif suite == "text":
        suite = TestSuite(TEXT_TESTS)
    elif suite == "preprocessing":
        suite = TestSuite(PREPROCESSING_TESTS)
    elif suite == "model":
        suite = TestSuite(MODEL_TESTS)
    elif suite == "dev":
        suite = TestSuite(DEV_TESTS)
    runner = TextTestRunner(verbosity=3)
    if isinstance(suite, str):
        logger.error("Please specify a test suite to run: i.e. 'dev' or 'all'")
    else:
        return runner.run(suite).wasSuccessful()


if __name__ == "__main__":
    try:
        suite = sys.argv[1]
    except IndexError:
        logger.info('No test suite specified, defaulting to "dev"')
        suite = "dev"
    result = run_tests(suite)
    if not result:
        sys.exit(1)
