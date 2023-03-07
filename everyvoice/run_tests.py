#!/usr/bin/env python3

""" Organize tests into Test Suites
"""

import os
import sys
from unittest import TestLoader, TestSuite, TextTestRunner

from loguru import logger

from everyvoice.tests.test_configs import ConfigTest
from everyvoice.tests.test_dataloader import DataLoaderTest
from everyvoice.tests.test_model import ModelTest
from everyvoice.tests.test_preprocessing import PreprocessingTest
from everyvoice.tests.test_text import TextTest

# Unit tests


LOADER = TestLoader()

CONFIG_TESTS = [LOADER.loadTestsFromTestCase(test) for test in [ConfigTest]]

DATALOADER_TESTS = [LOADER.loadTestsFromTestCase(test) for test in [DataLoaderTest]]

TEXT_TESTS = [LOADER.loadTestsFromTestCase(test) for test in [TextTest]]

PREPROCESSING_TESTS = [
    LOADER.loadTestsFromTestCase(test) for test in [PreprocessingTest]
]

MODEL_TESTS = [LOADER.loadTestsFromTestCase(test) for test in [ModelTest]]


DEV_TESTS = (
    CONFIG_TESTS + DATALOADER_TESTS + TEXT_TESTS + PREPROCESSING_TESTS + MODEL_TESTS
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
        return runner.run(suite)


if __name__ == "__main__":
    try:
        run_tests(sys.argv[1])
    except IndexError:
        logger.error("Please specify a test suite to run: i.e. 'dev' or 'all'")
