#!/usr/bin/env python

import tempfile
from pathlib import Path
from unittest import TestCase, main

import yaml
from anytree import PreOrderIter

from everyvoice.tests.stubs import capture_stdout, temp_chdir
from everyvoice.tests.test_wizard import CONTACT_INFO_STATE
from everyvoice.wizard import StepNames as SN
from everyvoice.wizard.basic import ConfigFormatStep
from everyvoice.wizard.tour import State
from everyvoice.wizard.utils import EnumDict, NodeMixinWithNavigation


class WavFileDirectoryRelativePathTest(TestCase):
    """
    Make sure the wav files directory path is correctly handle when transformed
    to a relative path.
    """

    data_dir = Path(__file__).parent / "data"

    def setUp(self):
        """
        Create a mock state instead of doing all prior steps to ConfigFormatStep.
        """
        state = State(
            {
                SN.output_step.value: "John/Smith",
                SN.name_step.value: "Unittest",
                "dataset_0": State(
                    {
                        SN.dataset_name_step.value: "unit",
                        SN.wavs_dir_step.value: "Common-Voice",
                        SN.symbol_set_step.value: {
                            "characters": [
                                "A",
                                "D",
                                "E",
                                "H",
                                "I",
                                "J",
                                "K",
                            ]
                        },
                        "filelist_data": [
                            {
                                "text": "Sentence 1",
                                "basename": "5061f5c3-3bf9-42c6-a268-435c146efaf6/dd50ed81b889047cb4399e34b650a91fcbd3b2a5e36cf0068251d64274bffb61",
                                "language": "und",
                                "speaker": "default",
                            },
                            {
                                "text": "Sentence 2",
                                "basename": "5061f5c3-3bf9-42c6-a268-435c146efaf6/6c45ab8c6e2454142c95319ca37f7e4ff6526dddbcc7fc540572e4e53264ec47",
                                "language": "und",
                                "speaker": "default",
                            },
                            {
                                "text": "Sentence 3",
                                "basename": "5061f5c3-3bf9-42c6-a268-435c146efaf6/3947ae033faeb793e00f836648e240bc91c821798bccc76656ad3e7030b38878",
                                "language": "und",
                                "speaker": "default",
                            },
                            {
                                "text": "Sentence 4",
                                "basename": "5061f5c3-3bf9-42c6-a268-435c146efaf6/65b61440f9621084a1a1d8c461d177c765fad3aff91e0077296081931929629b",
                                "language": "und",
                                "speaker": "default",
                            },
                            {
                                "text": "Sentence 5",
                                "basename": "5061f5c3-3bf9-42c6-a268-435c146efaf6/8a124117481eaf8f91d23aa3acda301e7fae7de85e98c016383381d54a3d5049",
                                "language": "und",
                                "speaker": "default",
                            },
                        ],
                        "sox_effects": [["channel", "1"]],
                    }
                ),
            }
        )
        self.config = ConfigFormatStep()
        self.config.response = "yaml"
        self.config._state = state

    def test_wav_file_directory_local(self):
        """
        output directory is `.`
        wav files directory located in `.`
        """
        self.config.state[SN.output_step.value] = "."
        self.config.state[SN.name_step.value] = "Unittest"
        self.config.state.update(CONTACT_INFO_STATE)
        with capture_stdout():
            with tempfile.TemporaryDirectory() as tmpdir:
                with temp_chdir(tmpdir):
                    tmpdir = Path(tmpdir).absolute()
                    self.config.effect()
                    data_file = (
                        Path(self.config.state[SN.name_step.value])
                        / "config/everyvoice-shared-data.yaml"
                    )
                    with data_file.open(encoding="utf8") as fin:
                        config = yaml.load(fin, Loader=yaml.FullLoader)
        # Unittest/config/everyvoice-shared-data.yaml
        # Common-Voice/
        self.assertEqual(
            Path(config["source_data"][0]["data_dir"]), Path("../../Common-Voice")
        )

    def test_wav_file_directory_under_wavs_directory(self):
        """
        output directory is `.`
        wav files directory located in `wavs/`
        """
        self.config.state[SN.output_step.value] = "."
        self.config.state[SN.name_step.value] = "Unittest"
        self.config.state.update(CONTACT_INFO_STATE)
        wavs_dir = "wavs/Common-Voice"
        self.config.state["dataset_0"][SN.wavs_dir_step.value] = wavs_dir
        with capture_stdout():
            with tempfile.TemporaryDirectory() as tmpdir:
                with temp_chdir(tmpdir):
                    tmpdir = Path(tmpdir).absolute()
                    self.config.effect()
                    data_file = (
                        Path(self.config.state[SN.name_step.value])
                        / "config/everyvoice-shared-data.yaml"
                    )
                    with data_file.open(encoding="utf8") as fin:
                        config = yaml.load(fin, Loader=yaml.FullLoader)
        # Unittest/config/everyvoice-shared-data.yaml
        # wavs/Common-Voice/
        self.assertEqual(
            Path(config["source_data"][0]["data_dir"]), Path("../..") / wavs_dir
        )

    def test_output_not_local_and_wav_file_directory_local(self):
        """
        output directory is NOT `.`
        wav files directory located in `.`
        """
        self.config.state[SN.output_step.value] = "John/Smith"
        self.config.state[SN.name_step.value] = "Unittest"
        self.config.state.update(CONTACT_INFO_STATE)
        with capture_stdout():
            with tempfile.TemporaryDirectory() as tmpdir:
                with temp_chdir(tmpdir):
                    tmpdir = Path(tmpdir).absolute()
                    self.config.effect()
                    data_file = (
                        Path(self.config.state[SN.output_step.value])
                        / self.config.state[SN.name_step.value]
                        / "config/everyvoice-shared-data.yaml"
                    )
                    with data_file.open(encoding="utf8") as fin:
                        config = yaml.load(fin, Loader=yaml.FullLoader)
        # John/Smith/Unittest/config/everyvoice-shared-data.yaml
        # Common-Voice/
        self.assertEqual(
            Path(config["source_data"][0]["data_dir"]), Path("../../../../Common-Voice")
        )

    def test_output_not_local_and_wav_file_directory_under_hierarchy(self):
        """
        output directory is NOT `.`
        wav files directory located in `wavs/`
        """
        self.config.state[SN.output_step.value] = "John/Smith"
        self.config.state[SN.name_step.value] = "Unittest"
        self.config.state.update(CONTACT_INFO_STATE)
        wavs_dir = "wavs/Common-Voice"
        self.config.state["dataset_0"][SN.wavs_dir_step.value] = wavs_dir
        with capture_stdout():
            with tempfile.TemporaryDirectory() as tmpdir:
                with temp_chdir(tmpdir):
                    tmpdir = Path(tmpdir).absolute()
                    self.config.effect()
                    data_file = (
                        Path(self.config.state[SN.output_step.value])
                        / self.config.state[SN.name_step.value]
                        / "config/everyvoice-shared-data.yaml"
                    )
                    with data_file.open(encoding="utf8") as fin:
                        config = yaml.load(fin, Loader=yaml.FullLoader)
        # John/Smith/Unittest/config/everyvoice-shared-data.yaml
        # wavs/Common-Voice/
        self.assertEqual(
            Path(config["source_data"][0]["data_dir"]),
            Path("../../../..") / wavs_dir,
        )

    def test_absolute_wav_file_directory_and_local_experiment(self):
        """
        output directory is `.`
        wav files directory located in `/ABSOLUTE/wavs/`
        """
        self.config.state[SN.output_step.value] = "."
        self.config.state[SN.name_step.value] = "Unittest"
        self.config.state.update(CONTACT_INFO_STATE)
        with capture_stdout():
            with tempfile.TemporaryDirectory() as tmpdir:
                with temp_chdir(tmpdir):
                    tmpdir = Path(tmpdir).absolute()
                    wavs_dir = tmpdir / "wavs/Common-Voice"
                    self.config.state["dataset_0"][SN.wavs_dir_step.value] = wavs_dir
                    self.config.state["dataset_0"][SN.text_processing_step] = (0,)
                    self.config.effect()
                    data_file = (
                        Path(self.config.state[SN.name_step.value])
                        / "config/everyvoice-shared-data.yaml"
                    )
                    with data_file.open(encoding="utf8") as fin:
                        config = yaml.load(fin, Loader=yaml.FullLoader)
        # Unittest/config/everyvoice-shared-data.yaml
        # /tmpdir/wavs/Common-Voice/
        self.assertEqual(
            Path(config["source_data"][0]["data_dir"]),
            wavs_dir,
        )

    def test_absolute_wav_file_directory_and_nested_experiment(self):
        """
        output directory is NOT `.`
        wav files directory located in `/ABSOLUTE/wavs/`
        """
        self.config.state[SN.output_step.value] = "John/Smith"
        self.config.state[SN.name_step.value] = "Unittest"
        self.config.state.update(CONTACT_INFO_STATE)
        with capture_stdout():
            with tempfile.TemporaryDirectory() as tmpdir:
                with temp_chdir(tmpdir):
                    tmpdir = Path(tmpdir).absolute()
                    wavs_dir = tmpdir / "wavs/Common-Voice"
                    self.config.state["dataset_0"][SN.wavs_dir_step.value] = wavs_dir
                    self.config.state["dataset_0"][SN.text_processing_step] = tuple()
                    self.config.effect()
                    data_file = (
                        Path(self.config.state[SN.output_step.value])
                        / self.config.state[SN.name_step.value]
                        / "config/everyvoice-shared-data.yaml"
                    )
                    with data_file.open(encoding="utf8") as fin:
                        config = yaml.load(fin, Loader=yaml.FullLoader)
        # John/Smith/Unittest/config/everyvoice-shared-data.yaml
        # /tmpdir/wavs/Common-Voice/
        self.assertEqual(
            Path(config["source_data"][0]["data_dir"]),
            wavs_dir,
        )


class TestEnumDict(TestCase):
    """Test the EnumDict class"""

    def test_enum_dict(self):
        """Enum values need to behave the same with or without .value"""
        d = EnumDict()
        d[SN.audio_config_step] = "foo"
        self.assertEqual(d[SN.audio_config_step.value], "foo")
        self.assertEqual(d.get(SN.audio_config_step.value), "foo")

        d[SN.wavs_dir_step.value] = "bar"
        self.assertEqual(d[SN.wavs_dir_step], "bar")
        self.assertEqual(d.get(SN.wavs_dir_step), "bar")

        self.assertEqual(d.get(SN.filelist_format_step, None), None)
        self.assertEqual(d.get(SN.filelist_format_step.value, None), None)

        d.update({SN.contact_email_step: "a@b.com"})
        self.assertEqual(d[SN.contact_email_step.value], "a@b.com")

        self.assertEqual(
            d,
            {
                SN.audio_config_step.value: "foo",
                SN.wavs_dir_step.value: "bar",
                SN.contact_email_step.value: "a@b.com",
            },
        )


class Node(NodeMixinWithNavigation):
    def __init__(self, name, parent=None):
        super().__init__()
        self.name = name
        self.parent = parent


class TestNodeMixin(TestCase):
    def test_node_mixin(self):
        """Test the NodeMixinWithNavigation class"""
        root = Node("root")
        n1 = Node("n1", parent=root)
        _ = Node("n1:1", parent=n1)
        _ = Node("n1:2", parent=n1)
        n2 = Node("n2", parent=root)
        n21 = Node("n2:1", parent=n2)
        n211 = Node("n2:1:1", parent=n21)
        _ = Node("n2:1:1:1", parent=n211)
        _ = Node("n3", parent=root)

        forward_order = list(PreOrderIter(root))
        for prev, next in zip(forward_order, forward_order[1:] + [None]):
            self.assertEqual(prev.next(), next)

        for next, prev in zip(forward_order, [None] + forward_order[:-1]):
            self.assertEqual(next.prev(), prev)


if __name__ == "__main__":
    main()
