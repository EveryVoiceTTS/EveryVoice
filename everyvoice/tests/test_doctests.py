#!/usr/bin/env python

import doctest
import sys

from pytest import main

import everyvoice.demo.app
import everyvoice.text
import everyvoice.utils
import everyvoice.wizard.utils


def test_run_all_doctests(subtests) -> None:
    for module_with_doctests in (
        everyvoice.demo.app,
        everyvoice.text.features,
        everyvoice.text.text_processor,
        everyvoice.text.utils,
        everyvoice.utils,
        everyvoice.wizard.utils,
    ):
        with subtests.test("Running doctests in", module=module_with_doctests.__name__):
            results = doctest.testmod(module_with_doctests)
            assert not results.failed, results


if __name__ == "__main__":
    main(sys.argv)
