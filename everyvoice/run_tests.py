#!/usr/bin/env python

"""Organize tests into Test Suites"""

import argparse
import io
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from loguru import logger

# Unit tests

SUBMODULE_SUITES: dict[str, tuple[str, ...]] = {
    "fs2": ("/model/feature_prediction/FastSpeech2_lightning/fs2/tests/",),
    "wav2vec2aligner": ("/model/aligner/wav2vec2aligner/aligner/tests/",),
}
SUITES: dict[str, tuple[str, ...]] = {
    "all": (),  # relies on discovery for collection
    "config": ("test_configs",),
    "loader": ("test_dataloader",),
    "text": ("test_text", "test_utils", "test_doctests"),
    "preprocessing": ("test_preprocessing",),
    "model": ("test_model",),
    "cli": (
        "test_wizard",
        "test_cli",
        "test_wizard_helpers",
        "test_subsample",
        "test_custom_g2p",
    ),
    "evaluation": ("test_evaluation",),
    **SUBMODULE_SUITES,
}
dev_suites = (
    "config",
    "loader",
    "text",
    "preprocessing",
    "model",
    "cli",
    "evaluation",
    "fs2",
    "wav2vec2aligner",
)
SUITE_NAMES = ["all", "dev"] + sorted(SUITES.keys())
SUITES["dev"] = sum((SUITES[suite] for suite in dev_suites), start=())


class PytestCollectorPlugin:
    def __init__(self):
        self.collected = []

    def pytest_collection_modifyitems(self, session, config, items):
        self.collected.extend([item.nodeid for item in items])


def list_tests(suite: Sequence[str]):
    plugin = PytestCollectorPlugin()
    pytest_args = ["--collect-only", *suite, "-q"]
    with redirect_stdout(io.StringIO()):
        pytest.main(pytest_args, plugins=[plugin])
    return plugin.collected


def describe_suite(suite_name, suite_filenames: Sequence[str]):
    full_list = list_tests([])
    requested_list = list_tests(suite_filenames)
    requested_set = set(requested_list)
    print(f"Test suite '{suite_name}' includes:", *sorted(requested_list), sep="\n")
    print(
        f"\nTest suite '{suite_name}' excludes:",
        *sorted(test for test in full_list if test not in requested_set),
        sep="\n",
    )
    print(
        "\nTotal test cases",
        f"found: {len(full_list)};",
        f"included: {len(requested_list)};",
        f"excluded: {len(full_list) - len(requested_list)}.",
    )


def run_tests(suite: str, describe=False, verbose=False, no_capture=False):
    """Run the specified test suite."""
    logger.info(f"Loading test suite '{suite}'. This may take a while...")
    if suite not in SUITES:
        logger.error(f"Please specify a test suite to run: one of '{SUITE_NAMES}'.")
        return False

    test_suite = SUITES[suite]
    root_dir = Path(__file__).parent
    test_suite_filenames: list[str] = [
        str(
            root_dir / test_file[1:]
            if test_file.startswith("/")
            else root_dir / "tests" / f"{test_file}.py"
        )
        for test_file in test_suite
    ]
    # print(test_suite_filenames)
    if describe:
        describe_suite(suite, test_suite_filenames)
        return True
    else:
        pytest_args = ["--verbose"] if verbose else []
        if no_capture:
            pytest_args.append("--capture=no")
        return 0 == pytest.main([*test_suite_filenames, *pytest_args])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EveryVoice test suites.")
    parser.add_argument(
        "--no-capture", "-s", action="store_true", help="let all logs go to screen"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="show test names as they run"
    )
    parser.add_argument(
        "--describe", action="store_true", help="describe the selected test suite"
    )
    parser.add_argument(
        "suite",
        nargs="?",
        default="dev",
        help="the test suite to run [dev]",
        choices=SUITE_NAMES,
    )
    args = parser.parse_args()
    if args.no_capture:
        import everyvoice.tests.stubs as stubs

        stubs.VERBOSE_OVERRIDE = True

    result = run_tests(args.suite, args.describe, args.verbose, args.no_capture)
    if not result:
        sys.exit(1)


if __name__ == "__main__":
    main()
