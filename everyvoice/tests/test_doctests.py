#!/usr/bin/env python

import doctest
from unittest import TestCase, main

import everyvoice.demo.app
import everyvoice.text
import everyvoice.utils


class RunDocTests(TestCase):

    def test_run_all_doctests(self):
        for module_with_doctests in (
            everyvoice.demo.app,
            everyvoice.text.features,
            everyvoice.text.text_processor,
            everyvoice.text.utils,
            everyvoice.utils,
        ):
            with self.subTest(
                "Running doctests in", module=module_with_doctests.__name__
            ):
                results = doctest.testmod(module_with_doctests)
                self.assertFalse(results.failed, results)


if __name__ == "__main__":
    main()
