#!/usr/bin/env python
import doctest
from pathlib import Path
from unittest import TestCase

from pep440 import is_canonical
from pydantic import BaseModel
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated

import everyvoice.utils
from everyvoice._version import VERSION
from everyvoice.config.shared_types import init_context
from everyvoice.utils import (
    directory_path_must_exist,
    path_is_a_directory,
    relative_to_absolute_path,
)


class VersionTest(TestCase):
    def test_version_is_pep440_compliant(self):
        self.assertTrue(is_canonical(VERSION))


class UtilsTest(TestCase):
    def test_run_doctest(self):
        """Run doctests in everyvoice.utils"""
        results = doctest.testmod(everyvoice.utils)
        self.assertFalse(results.failed, results)


class PathIsADirectory(BaseModel):
    """Dummy Class for PathIsADirectoryTest"""

    path: Annotated[Path, BeforeValidator(path_is_a_directory)]


class PathIsADirectoryTest(TestCase):
    """Testing when we Annotated with path_is_a_directory"""

    def test_using_a_directory(self):
        """
        Verifies that MustBeDir detects that the argument is a directory.
        """
        try:
            root_dir = Path(__file__).parent / "data"
            root_dir = root_dir.resolve()
            directory = Path("hierarchy")
            self.assertTrue((root_dir / directory).exists())
            PathIsADirectory(path=root_dir / directory)
        except ValueError:
            self.fail("Failed to detect that the argument is a directory")

    def WIP_test_using_a_directory_with_context(self):
        """
        Verifies that MustBeDir detects that the argument is a directory.
        """
        # FIXME: for some strange reason, the context is not getting populated.
        try:
            root_dir = Path(__file__).parent / "data"
            root_dir = root_dir.resolve()
            directory = Path("hierarchy")
            self.assertTrue((root_dir / directory).exists())
            with init_context({"writing_config": root_dir.parent.resolve()}):
                PathIsADirectory(path=directory)
        except ValueError:
            self.fail("Failed to detect that the argument is a directory")

    def test_using_a_file(self):
        """
        Verifies that MustBeDir detects that the argument is a file.
        """
        with self.assertRaises(ValueError):
            PathIsADirectory(path=Path(__file__))

    def WIP_test_using_a_file_with_context(self):
        """
        Verifies that MustBeDir detects that the argument is a file.
        """
        # FIXME: for some strange reason, the context is not getting populated.
        with self.assertRaises(ValueError):
            with init_context({"writing_config": Path(__file__).parent.resolve()}):
                PathIsADirectory(path=Path(__file__))


class RelativePathToAbsolute(BaseModel):
    """Dummy Class for RelativePathToAbsoluteTest"""

    path: Annotated[Path, BeforeValidator(relative_to_absolute_path)]


class RelativePathToAbsoluteTest(TestCase):
    """Testing when we Annotated with relative_to_absolute_path"""

    def test_should_not_change(self):
        """
        Without context, the path should stay the same.
        """
        path = Path(__file__).parent / "data"
        path = path.relative_to(Path(__file__).parent)
        test = RelativePathToAbsolute(path=path)
        self.assertEqual(test.path, path)

    def WIP_test_with_context(self):
        """
        When provided with a context, the path should be absolute.
        """
        # FIXME: for some strange reason, the context is not getting populated.
        root_dir = Path(__file__).parent.resolve()
        path = Path("data")
        with init_context({"config_path": root_dir}):
            test = RelativePathToAbsolute(path=path)
            print(f"{test=}")
            self.assertTrue(test.path.is_absolute())

    def test_invalid_entry(self):
        """
        If the provide type cannot be a path, we should fail.
        """
        with self.assertRaises(ValueError):
            RelativePathToAbsolute(path=4)


class DirectoryPathMustExist(BaseModel):
    """Dummy Class for DirectoryPathMustExistTest"""

    path: Annotated[Path, BeforeValidator(directory_path_must_exist)]


class DirectoryPathMustExistTest(TestCase):
    """Testing when we Annotated with directory_path_must_exist"""

    def test_using_a_directory(self):
        """
        Verifies that MustBeDir detects that the argument is a directory.
        """
        try:
            DirectoryPathMustExist(path=Path(__file__).parent / "data")
        except ValueError:
            self.fail("Failed to detect that the path exists")

    def test_using_a_directory_with_context(self):
        """
        Verifies that MustBeDir detects that the argument is a directory.
        """
        try:
            root_dir = Path(__file__).parent / "data"
            root_dir = root_dir.resolve()
            filename = Path("metadata.csv")
            self.assertTrue((root_dir / filename).exists())
            with init_context({"writing_config": root_dir.resolve()}):
                DirectoryPathMustExist(path=filename)
        except ValueError:
            self.fail("Failed to detect that the path exists")
