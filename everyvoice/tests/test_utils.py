#!/usr/bin/env python
import doctest
from unittest import TestCase, main

import everyvoice.utils


class UtilsTest(TestCase):
    def test_run_doctest(self):
        """Run doctests in everyvoice.utils"""
        results = doctest.testmod(everyvoice.utils)
        self.assertFalse(results.failed, results)


if __name__ == "__main__":
    main()
