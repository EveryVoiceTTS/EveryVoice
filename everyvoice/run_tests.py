#!/usr/bin/env python

""" Organize tests into Test Suites
"""

import importlib
import os
import re
import sys
from typing import Iterable
from unittest import TestLoader, TestSuite, TextTestRunner

from loguru import logger

# Unit tests

SUBMODULE_SUITES: dict[str, tuple[str, ...]] = {
    "fs2": ("/model/feature_prediction/FastSpeech2_lightning/fs2/tests",),
}
SUITES: dict[str, tuple[str, ...]] = {
    "config": ("test_configs",),
    "loader": ("test_dataloader",),
    "text": ("test_text", "test_utils"),
    "preprocessing": ("test_preprocessing",),
    "model": ("test_model",),
    "cli": ("test_wizard", "test_cli"),
    **SUBMODULE_SUITES,
}
dev_suites = ("config", "loader", "text", "preprocessing", "model", "cli", "fs2")
SUITES["dev"] = sum((SUITES[suite] for suite in dev_suites), start=())


def remove_test_prefix(test_case: str):
    for prefix in "<", "everyvoice.", "tests.":
        if test_case.startswith(prefix):
            test_case = test_case[len(prefix) :]
    return "<" + test_case


def list_tests(suite: TestSuite):
    for subsuite in suite:
        # print(str(subsuite))
        for match in re.finditer(r"tests=\[([^][]+)\]>", str(subsuite)):
            for test_case in match[1].split(", "):
                yield remove_test_prefix(test_case)


def all_test_suites() -> TestSuite:
    loader = TestLoader()
    # NOTE: Looking specifically under `/tests` removes empty TestSuites.
    test_suite = loader.discover(
        os.path.dirname(__file__) + "/tests",
        top_level_dir=os.path.dirname(__file__),
    )
    for submodule_testsuite in SUBMODULE_SUITES.values():
        suite = loader.discover(
            os.path.dirname(__file__) + submodule_testsuite[0],
        )
        test_suite.addTests(suite)

    return test_suite


def describe_suite(suite: TestSuite):
    full_suite = all_test_suites()
    full_list = list(list_tests(full_suite))
    requested_list = list(list_tests(suite))
    requested_set = set(requested_list)
    print("Test suite includes:", *sorted(requested_list), sep="\n")
    print(
        "\nTest suite excludes:",
        *sorted(test for test in full_list if test not in requested_set),
        sep="\n",
    )


def run_tests(suite: str, describe: bool = False):
    """Decide which Test Suite to run"""
    logger.info(f"Loading test suite '{suite}'. This may take a while...")
    if suite == "all":
        test_suite = all_test_suites()
    else:
        loader = TestLoader()
        tests: Iterable[str]
        if suite in SUITES:
            tests = SUITES[suite]
        else:
            logger.error(
                f"Please specify a test suite to run: one of '{['all'] + sorted(SUITES.keys())}'."
            )
            return False
        tests = [
            "everyvoice.tests." + test if not test.startswith("/") else test
            for test in tests
        ]
        test_suite = TestSuite()
        for test in tests:
            logger.info(f"Loading {test=}")
            if test.startswith("/"):
                sub_suite = loader.discover(
                    os.path.dirname(__file__) + test,
                    top_level_dir=os.path.dirname(__file__),  # MANDATORY
                )
                test_suite.addTests(sub_suite)
            else:
                importlib.import_module(test)
                test_suite.addTest(loader.loadTestsFromName(test))

    if describe:
        describe_suite(test_suite)
        return True
    else:
        logger.info("Running test suite")
        return TextTestRunner(verbosity=3).run(test_suite).wasSuccessful()


if __name__ == "__main__":
    describe = "--describe" in sys.argv
    if describe:
        sys.argv.remove("--describe")

    try:
        suite = sys.argv[1]
    except IndexError:
        logger.info('No test suite specified, defaulting to "dev"')
        suite = "dev"
    result = run_tests(suite, describe)
    if not result:
        sys.exit(1)
