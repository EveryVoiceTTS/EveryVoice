import doctest
import tempfile
from pathlib import Path
from typing import Any
from unittest import TestCase

import torch
from pep440 import is_canonical
from pydantic import BaseModel
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated

import everyvoice.utils
from everyvoice._version import VERSION
from everyvoice.config.shared_types import _init_context_var, init_context
from everyvoice.config.validation_helpers import (
    directory_path_must_exist,
    path_is_a_directory,
    relative_to_absolute_path,
)
from everyvoice.tests.stubs import capture_logs
from everyvoice.utils import write_filelist
from everyvoice.utils.heavy import get_device_from_accelerator


class VersionTest(TestCase):
    def test_version_is_pep440_compliant(self):
        self.assertTrue(is_canonical(VERSION))


class UtilsTest(TestCase):
    def test_run_doctest(self):
        """Run doctests in everyvoice.utils"""
        results = doctest.testmod(everyvoice.utils)
        self.assertFalse(results.failed, results)

    def test_write_filelist(self):
        """Filelist should write files with headers in order"""
        basic_files = [
            {
                "basename": "test",
                "phones": "foo",
                "characters": "bar",
                "language": "test",
                "extra": "test",
            }
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            basic_path = tempdir / "test.psv"
            write_filelist(basic_files, basic_path)
            with open(basic_path) as f:
                headers = f.readline().strip().split("|")
            self.assertEqual(len(headers), 5)
            self.assertEqual(headers[0], "basename")
            self.assertEqual(headers[1], "language")
            self.assertEqual(headers[2], "characters")
            self.assertEqual(headers[3], "phones")
            self.assertEqual(headers[4], "extra")


class ContextableBaseModel(BaseModel):
    """
    Enable using
       with init_context({"k":v}):
          pass
    which enable passing a context down to a pydantic BaseModel.
    """

    # [Using validation context with BaseModel initialization](https://docs.pydantic.dev/2.3/usage/validators/#using-validation-context-with-basemodel-initialization)
    def __init__(__pydantic_self__, **data: Any) -> None:
        __pydantic_self__.__pydantic_validator__.validate_python(
            data,
            self_instance=__pydantic_self__,
            context=_init_context_var.get(),
        )


class PathIsADirectory(ContextableBaseModel):
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


class RelativePathToAbsolute(ContextableBaseModel):
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
        If the provided type cannot be a path, we should fail.
        """
        with self.assertRaises(ValueError):
            RelativePathToAbsolute(path=4)


class DirectoryPathMustExist(ContextableBaseModel):
    """Dummy Class for DirectoryPathMustExistTest"""

    path: Annotated[Path, BeforeValidator(directory_path_must_exist)]


class DirectoryPathMustExistTest(TestCase):
    """
    Testing when we Annotated with directory_path_must_exist.
    It should create a directory if it doesn't exist.
    """

    def test_path_already_exists(self):
        path = Path(__file__).parent / "data"
        with capture_logs() as output:
            dir = DirectoryPathMustExist(path=path)
            # TODO: Should check that there is no log produced saying a directory was created
            self.assertListEqual(output, [])
        self.assertTrue(dir.path.exists())

    def test_using_a_directory(self):
        """
        Automatically create a directory.
        """
        with tempfile.TemporaryDirectory() as tmpdir, capture_logs() as output:
            path = Path(tmpdir) / "test_using_a_directory"
            self.assertFalse(path.exists())
            dir = DirectoryPathMustExist(path=path)
            self.assertEqual(dir.path, path)
            self.assertIn(
                f"Directory at {path} does not exist. Creating...",
                output[0],
            )
            self.assertTrue(path.exists())
            self.assertTrue(dir.path.exists())

    def test_using_a_directory_with_context(self):
        """
        Verifies that directory_path_must_exist(), when using a context,
        creates the directory if it doesn't exist.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            directory = Path("test_using_a_directory_with_context")
            self.assertFalse((root_dir / directory).exists())
            with init_context({"writing_config": root_dir.resolve()}):
                dir = DirectoryPathMustExist(path=directory)
            # Note: dir.path shouldn't not change to an absolute value.
            self.assertEqual(dir.path, directory)
            self.assertTrue((root_dir / directory).exists())
            # Note: since dir.path is NOT replaced with an absolute it
            # shouldn't exist because it was created relative to the context's
            # path.
            self.assertFalse(dir.path.exists())


class GetDeviceFromAcceleratorTest(TestCase):
    def test_auto(self):
        self.assertEqual(
            get_device_from_accelerator("auto"),
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )

    def test_cpu(self):
        self.assertEqual(get_device_from_accelerator("cpu"), torch.device("cpu"))

    def test_gpu(self):
        self.assertEqual(get_device_from_accelerator("gpu"), torch.device("cuda:0"))

    def test_mps(self):
        self.assertEqual(get_device_from_accelerator("mps"), torch.device("mps"))

    def test_unknown_accelerator(self):
        self.assertEqual(get_device_from_accelerator("unknown"), torch.device("cpu"))
