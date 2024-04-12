"""
Common base class for the EveryVoice test suites

Adapted from https://github.com/ReadAlongs/Studio/blob/main/test/basic_test_case.py

"""

from pathlib import Path
from unittest import TestCase

from everyvoice.config.shared_types import ContactInformation


class BasicTestCase(TestCase):
    """A Basic Unittest build block class that comes bundled with
    the path to the test data (self.data_dir)

    For convenience, self.data_dir is pathlib.Path objects that can be used
    either with os.path functions or the shorter Path operators.
    E.g., these two lines are equivalent:
        text_file = os.path.join(self.data_dir, "ej-fra.txt")
        text_file = self.data_dir / "ej-fra.txt"
    """

    data_dir = Path(__file__).parent / "data"
    contact = ContactInformation(
        contact_name="Test Runner", contact_email="info@everyvoice.ca"
    )
