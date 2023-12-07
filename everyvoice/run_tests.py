#!/usr/bin/env python

""" Organize tests into Test Suites
"""

import importlib
import os
import sys
from unittest import TestLoader, TestSuite, TextTestRunner

from loguru import logger

# Unit tests

SUITES = {
    "config": ("test_configs",),
    "loader": ("test_dataloader",),
    "text": ("test_text", "test_utils"),
    "preprocessing": ("test_preprocessing",),
    "model": ("test_model",),
    "cli": ("test_wizard", "test_cli"),
}
dev_suites = ("config", "loader", "text", "preprocessing", "model", "cli")
SUITES["dev"] = sum((SUITES[suite] for suite in dev_suites), start=())


def run_tests(suite):
    """Decide which Test Suite to run"""
    loader = TestLoader()
    logger.info(f"Loading test suite '{suite}'.")
    if suite == "all":
        suite = loader.discover(os.path.dirname(__file__))
    else:
        if suite in SUITES:
            tests = SUITES[suite]
        else:
            logger.error(
                f"Please specify a test suite to run: one of '{['all'] + sorted(SUITES.keys())}'."
            )
            return False
        tests = ["everyvoice.tests." + test for test in tests]
        for test in tests:
            importlib.import_module(test)
        suite = TestSuite(loader.loadTestsFromNames(tests))

    logger.info("Running test suite")
    return TextTestRunner(verbosity=3).run(suite).wasSuccessful()


if __name__ == "__main__":
    try:
        suite = sys.argv[1]
    except IndexError:
        logger.info('No test suite specified, defaulting to "dev"')
        suite = "dev"
    result = run_tests(suite)
    if not result:
        sys.exit(1)
