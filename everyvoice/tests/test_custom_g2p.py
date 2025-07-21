"""Unit tests for using custom g2p functions"""

import tempfile
from pathlib import Path
from textwrap import dedent
from unittest import TestCase

import yaml
from pydantic import ValidationError

import everyvoice
from everyvoice.config.text_config import TextConfig
from everyvoice.tests.stubs import (
    Say,
    mute_logger,
    null_patch,
    patch_input,
    patch_logger,
    patch_menu_prompt,
    patch_questionary,
)
from everyvoice.text.phonemizer import (
    AVAILABLE_G2P_ENGINES,
    DEFAULT_G2P,
    CachingG2PEngine,
    get_g2p_engine,
    make_default_g2p_engines,
)
from everyvoice.wizard import basic, dataset

from .test_wizard import RecursiveAnswers, StepAndAnswer, WizardTestBase


class TestG2p(TestCase):
    """Test G2P"""

    def setUp(self) -> None:
        super().setUp()
        # Each test in this suite needs to start with a fresh, empty
        # AVAILABLE_G2P_ENGINES cache, otherwise caching due to previous calls
        # get get_g2p_engine() in other suites could invalidate some tests here.
        self.SAVED_AVAILABLE_G2P_ENGINES = dict(AVAILABLE_G2P_ENGINES)
        AVAILABLE_G2P_ENGINES.clear()
        AVAILABLE_G2P_ENGINES.update(make_default_g2p_engines())

    def tearDown(self) -> None:
        super().setUp()
        AVAILABLE_G2P_ENGINES.clear()
        AVAILABLE_G2P_ENGINES.update(self.SAVED_AVAILABLE_G2P_ENGINES)

    def test_many_available_langs(self):
        self.assertGreaterEqual(len(AVAILABLE_G2P_ENGINES), 20)

    def test_pua_chars(self):
        eng_g2p = get_g2p_engine("eng")
        und_g2p = get_g2p_engine("und")
        tokens = eng_g2p("h_e_l_l_o")
        self.assertEqual(
            tokens,
            ["e", "ɪ", "t", "ʃ", "_", "i", "_", "ɛ", "l", "_", "ɛ", "l", "_", "o", "ʊ"],
        )
        tokens = und_g2p("___")
        self.assertEqual(tokens, ["_", "_", "_"])

    def test_basic_g2p(self):
        eng_g2p = get_g2p_engine("eng")
        self.assertEqual(
            eng_g2p("hello world"), ["h", "ʌ", "l", "o", "ʊ", " ", "w", "ɜ˞", "l", "d"]
        )
        # keep's punctuation
        self.assertEqual(
            eng_g2p('hello "world"!!?.'),
            [
                "h",
                "ʌ",
                "l",
                "o",
                "ʊ",
                " ",
                '"',
                "w",
                "ɜ˞",
                "l",
                "d",
                '"',
                "!",
                "!",
                "?",
                ".",
            ],
        )
        # another language
        str_g2p = get_g2p_engine("str")
        self.assertEqual(str_g2p("SENĆOŦEN"), ["s", "ʌ", "n", "t͡ʃ", "ɑ", "θ", "ʌ", "n"])
        # test lang_id missing
        with self.assertRaises(NotImplementedError):
            get_g2p_engine("boop")

    def test_phonemizer_normalization(self):
        moh_g2p = get_g2p_engine("moh")
        self.assertEqual(moh_g2p("\u00e9"), ["\u00e9"])

    def test_invalid_lang_id(self):
        """
        User asked for a language that is not supported by AVAILABLE_G2P_ENGINES.
        """
        lang_id = "unittest"
        self.assertNotIn(lang_id, AVAILABLE_G2P_ENGINES)
        with self.assertRaisesRegex(
            NotImplementedError,
            rf"Sorry, we don't have a grapheme-to-phoneme engine available for {lang_id}.*",
            msg="The user provided G2P engine shouldn't be available before loading a TextConfig.",
        ):
            get_g2p_engine(lang_id)

    def test_custom_g2p_engine(self):
        """
        Use a user provided G2P engine.
        """
        lang_id = "unittest"
        with self.assertRaisesRegex(
            NotImplementedError,
            rf"Sorry, we don't have a grapheme-to-phoneme engine available for {lang_id}.*",
            msg="The user provided G2P engine shouldn't be available before loading a TextConfig.",
        ):
            get_g2p_engine(lang_id)
        with mute_logger("everyvoice.config.text_config"):
            TextConfig(g2p_engines={lang_id: "everyvoice.tests.g2p_engines.valid"})
        self.assertIn(lang_id, AVAILABLE_G2P_ENGINES)
        self.assertIs(
            AVAILABLE_G2P_ENGINES[lang_id],
            everyvoice.tests.g2p_engines.valid,
        )

    def test_invalid_g2p_engine(self):
        """
        The only string value allowed in AVAILABLE_G2P_ENGINES is 'DEFAULT_G2P'.
        """

        lang_id = "unittest"
        AVAILABLE_G2P_ENGINES[lang_id] = "WRONG"
        with self.assertRaisesRegex(
            AssertionError,
            f"Internal error: the only str value allowed in AVAILABLE_G2P_ENGINES is '{DEFAULT_G2P}'.",
        ):
            get_g2p_engine(lang_id)

    def test_autoload(self):
        """
        Default G2PEngine should autoload a CachingG2PEngine(lang_id).
        """
        lang_id = "eng"
        self.assertIn(lang_id, AVAILABLE_G2P_ENGINES)
        self.assertEqual(AVAILABLE_G2P_ENGINES[lang_id], DEFAULT_G2P)

        g2p_engine = get_g2p_engine(lang_id)
        self.assertFalse(isinstance(g2p_engine, str))
        self.assertTrue(isinstance(g2p_engine, CachingG2PEngine))


class TextConfigWithG2pTest(TestCase):
    """
    TextConfig
    """

    def setUp(self) -> None:
        super().setUp()
        self.AVAILABLE_G2P_ENGINES = dict(AVAILABLE_G2P_ENGINES)

    def tearDown(self) -> None:
        super().setUp()
        AVAILABLE_G2P_ENGINES.clear()
        AVAILABLE_G2P_ENGINES.update(self.AVAILABLE_G2P_ENGINES)

    def test_no_user_provided_g2p_engines(self):
        """
        The TextConfig doesn't contain new g2p engines.
        """
        num_g2p_engines = len(AVAILABLE_G2P_ENGINES.keys())
        TextConfig()
        self.assertEqual(num_g2p_engines, len(AVAILABLE_G2P_ENGINES.keys()))

    def test_loading_g2p_engines(self):
        """
        Simulate user provided G2P engines.
        """

        lang_id_1, lang_id_2 = "unittest1", "unittest2"
        with mute_logger("everyvoice.config.text_config"):
            TextConfig(
                g2p_engines={
                    lang_id_1: "everyvoice.tests.g2p_engines.valid",
                    lang_id_2: "everyvoice.tests.g2p_engines.valid",
                }
            )
        self.assertIn(lang_id_1, AVAILABLE_G2P_ENGINES)
        self.assertIn(lang_id_2, AVAILABLE_G2P_ENGINES)
        self.assertIs(
            AVAILABLE_G2P_ENGINES[lang_id_1],
            everyvoice.tests.g2p_engines.valid,
        )
        self.assertIs(
            AVAILABLE_G2P_ENGINES[lang_id_2],
            everyvoice.tests.g2p_engines.valid,
        )

    def test_loading_g2p_engines_with_invalid_module(self):
        """
        Simulate user provided G2P engines module that doesn't exist.
        """

        lang_id = "unittest"
        with (
            self.assertRaisesRegex(
                ValueError,
                rf".*Invalid G2P engine module `unknown_module` for `{lang_id}`.*",
            ),
            patch_logger(everyvoice.config.text_config) as logger,
            self.assertLogs(logger) as logs,
        ):
            TextConfig(g2p_engines={lang_id: "unknown_module.g2p"})
        self.assertNotIn(lang_id, AVAILABLE_G2P_ENGINES)
        self.assertIn("Invalid G2P engine", "\n".join(logs.output))

    def test_g2p_engine_signature_multiple_arguments(self):
        """
        User provided a G2P function that takes too many arguments.
        """

        lang_id = "unittest"
        with self.assertRaisesRegex(
            ValidationError,
            r".*G2P engine's signature should take a single argument.*",
        ):
            TextConfig(
                g2p_engines={lang_id: "everyvoice.tests.g2p_engines.multiple_arguments"}
            )
        self.assertNotIn(lang_id, AVAILABLE_G2P_ENGINES)

    def test_g2p_engine_signature_not_a_string(self):
        """
        User provided a G2P engine that doesn't take a string as input.
        """
        lang_id = "unittest"
        with self.assertRaisesRegex(
            ValidationError,
            r".*G2P Engine's signature should take a string.*",
        ):
            TextConfig(
                g2p_engines={lang_id: "everyvoice.tests.g2p_engines.not_a_string"}
            )
        self.assertNotIn(lang_id, AVAILABLE_G2P_ENGINES)

    def test_g2p_engine_signature_not_a_list(self):
        """
        User provided a G2P engine that doesn't return a list of strings.
        """
        lang_id = "unittest"
        with self.assertRaisesRegex(
            ValidationError,
            r".*G2P Engine's signature should return a list of strings.*",
        ):
            TextConfig(g2p_engines={lang_id: "everyvoice.tests.g2p_engines.not_a_list"})
        self.assertNotIn(lang_id, AVAILABLE_G2P_ENGINES)

    def test_overriding_default_g2p_engine(self):
        """
        User provided a G2P engine that overrides a default G2P engine.
        """
        num_g2p_engines = len(AVAILABLE_G2P_ENGINES.keys())
        lang_id = "fra"
        self.assertIn(lang_id, AVAILABLE_G2P_ENGINES)
        old_g2p_engine = AVAILABLE_G2P_ENGINES[lang_id]
        with mute_logger("everyvoice.config.text_config"):
            TextConfig(g2p_engines={lang_id: "everyvoice.tests.g2p_engines.valid"})
        self.assertEqual(
            num_g2p_engines,
            len(AVAILABLE_G2P_ENGINES.keys()),
            "This shouldn't add a new G2P Engine.",
        )
        self.assertIs(
            AVAILABLE_G2P_ENGINES[lang_id],
            everyvoice.tests.g2p_engines.valid,
        )
        self.assertIsNot(
            old_g2p_engine,
            AVAILABLE_G2P_ENGINES[lang_id],
            "The new G2P Engine shouldn't be the same as the old engine.",
        )


class CustomG2pTest(WizardTestBase):
    DATASET1 = dedent(
        """\
        basename|text
        f1|foo bar
        f2|bar baz
        f3|baz foo
        """
    )

    DATASET2 = dedent(
        """\
        basename|language|text
        f4|str|foo foo
        f5|git|bar bar
        f6|und|baz baz
        f7|lang1|a b c
        f7|unknown-lang|zang
        """
    )

    basic_steps = [
        StepAndAnswer(basic.NameStep(), Say("project")),
        StepAndAnswer(basic.ContactNameStep(), Say("Test Name")),
        StepAndAnswer(basic.ContactEmailStep(), Say("info@everyvoice.ca")),
    ]

    def test_custom_g2p_in_wizard(self):
        """Test using a custom g2p on a custom language code"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with open(tmpdir / "filelist1.psv", "w", encoding="utf8") as f:
                f.write(self.DATASET1)
            with open(tmpdir / "filelist2.psv", "w", encoding="utf8") as f:
                f.write(self.DATASET2)
            for i in range(1, 8):
                with open(tmpdir / f"f{i}.wav", "wb"):
                    pass
            third_dataset_children_answers = [
                RecursiveAnswers(
                    patch_questionary(tmpdir / "filelist1.psv")
                ),  # filelist step
                RecursiveAnswers(
                    patch_menu_prompt(1),  # yes to permission
                    children_answers=[
                        RecursiveAnswers(patch_menu_prompt(0)),  # psv format
                        RecursiveAnswers(patch_menu_prompt(0)),  # characters
                        RecursiveAnswers(patch_menu_prompt(())),  # no text prepro
                        RecursiveAnswers(
                            patch_menu_prompt(0),  # no, does not have speaker column
                            children_answers=[
                                RecursiveAnswers(
                                    patch_menu_prompt(1),  # Yes, give speaker ID
                                    children_answers=[
                                        RecursiveAnswers(patch_input("my_speaker"))
                                    ],
                                ),
                            ],
                        ),
                        RecursiveAnswers(
                            patch_menu_prompt(0),  # no, there is no language column
                            children_answers=[
                                RecursiveAnswers(
                                    patch_menu_prompt(1),  # [custom] lang code option
                                    children_answers=[
                                        RecursiveAnswers(
                                            patch_input(
                                                ["not valid / bad slug", "my-lang"],
                                                multi=True,
                                            )
                                        )
                                    ],
                                )
                            ],
                        ),
                        # Keep g2p settings means this project will have one non-g2p-able lang
                        RecursiveAnswers(patch_menu_prompt(0)),  # Keep g2p settings
                        RecursiveAnswers(patch_questionary(tmpdir)),  # wav directory
                        RecursiveAnswers(null_patch()),  # ValidateWavsStep
                        RecursiveAnswers(null_patch()),  # SymbolSetStep
                        RecursiveAnswers(patch_menu_prompt([])),  # no Sox
                        RecursiveAnswers(patch_input("dataset2")),  # Dataset name
                    ],
                ),
                RecursiveAnswers(
                    patch_menu_prompt(0),  # no more data
                    children_answers=[RecursiveAnswers(patch_menu_prompt(0))],  # yaml
                ),
            ]
            second_dataset_children_answers = [
                RecursiveAnswers(
                    patch_questionary(tmpdir / "filelist2.psv")
                ),  # filelist step
                RecursiveAnswers(
                    patch_menu_prompt(1),  # yes to permission
                    children_answers=[
                        RecursiveAnswers(patch_menu_prompt(0)),  # psv format
                        RecursiveAnswers(patch_menu_prompt(0)),  # characters
                        RecursiveAnswers(patch_menu_prompt(())),  # no text prepro
                        RecursiveAnswers(
                            patch_menu_prompt(0),  # no, does not have speaker column
                            children_answers=[
                                RecursiveAnswers(
                                    patch_menu_prompt(1),  # Yes, give speaker ID
                                    children_answers=[
                                        RecursiveAnswers(patch_input("my_speaker"))
                                    ],
                                ),
                            ],
                        ),
                        RecursiveAnswers(
                            patch_menu_prompt(1),  # yes, there is a language column
                            children_answers=[
                                RecursiveAnswers(Say(1))  # speaker column is is 1
                            ],
                        ),
                        RecursiveAnswers(
                            patch_menu_prompt(2),  # custom g2p for "lang1"
                            children_answers=[
                                RecursiveAnswers(
                                    patch_input("everyvoice.tests.g2p_engines.valid")
                                ),
                                RecursiveAnswers(
                                    patch_menu_prompt(0),  # done with custom g2p
                                ),
                            ],
                        ),
                        RecursiveAnswers(patch_questionary(tmpdir)),  # wav directory
                        RecursiveAnswers(null_patch()),  # ValidateWavsStep
                        RecursiveAnswers(null_patch()),  # SymbolSetStep
                        RecursiveAnswers(patch_menu_prompt([])),  # no Sox
                        RecursiveAnswers(patch_input("dataset1")),  # Dataset name
                    ],
                ),
                RecursiveAnswers(
                    patch_menu_prompt(1),  # yes more data
                    children_answers=third_dataset_children_answers,
                ),
            ]
            steps_and_answers = [
                *self.basic_steps,
                StepAndAnswer(
                    basic.OutputPathStep(), patch_questionary(tmpdir / "out")
                ),
                # First dataset
                StepAndAnswer(
                    dataset.FilelistStep(state_subset="dataset_0"),
                    patch_questionary(tmpdir / "filelist1.psv"),
                ),
                StepAndAnswer(
                    dataset.FilelistFormatStep(state_subset="dataset_0"),
                    patch_menu_prompt(0),  # psv
                ),
                StepAndAnswer(
                    dataset.FilelistTextRepresentationStep(state_subset="dataset_0"),
                    patch_menu_prompt(0),  # characters
                ),
                StepAndAnswer(
                    dataset.TextProcessingStep(state_subset="dataset_0"),
                    patch_menu_prompt(()),
                ),
                StepAndAnswer(
                    dataset.HasSpeakerStep(state_subset="dataset_0"),
                    patch_menu_prompt(0),  # 0 is no
                    children_answers=[
                        RecursiveAnswers(patch_menu_prompt(0)),  # 0 is no
                    ],
                ),
                StepAndAnswer(
                    dataset.HasLanguageStep(state_subset="dataset_0"),
                    patch_menu_prompt(0),  # 0 is no
                    children_answers=[RecursiveAnswers(Say("git"))],
                ),
                StepAndAnswer(
                    dataset.CustomG2PStep(state_subset="dataset_0"),
                    patch_menu_prompt(1),  # custom g2p for "git"
                    children_answers=[
                        # RecursiveAnswers(Say("everyvoice.tests.g2p_engines.valid"))
                        RecursiveAnswers(
                            patch_input(
                                (
                                    "asdf",
                                    "everyvoice.tests.g2p_engines.not_a_list",
                                    "everyvoice.tests.g2p_engines.g2p_test_upper",
                                ),
                                multi=True,
                            )
                        ),
                        RecursiveAnswers(patch_menu_prompt(0)),  # keep g2p
                    ],
                ),
                StepAndAnswer(
                    dataset.WavsDirStep(state_subset="dataset_0"),
                    patch_questionary(tmpdir),
                ),
                StepAndAnswer(
                    dataset.SymbolSetStep(state_subset="dataset_0"),
                    null_patch(),
                ),
                StepAndAnswer(
                    dataset.SoxEffectsStep(state_subset="dataset_0"),
                    patch_menu_prompt([]),
                ),
                StepAndAnswer(
                    dataset.DatasetNameStep(state_subset="dataset_0"),
                    patch_input("dataset0"),
                ),
                StepAndAnswer(
                    basic.MoreDatasetsStep(),
                    patch_menu_prompt(1),  # 1 is yes
                    children_answers=second_dataset_children_answers,
                ),
            ]
            tour, _ = self.monkey_run_tour(
                "Tour with three datasets and some custom g2p functions, valid and not",
                steps_and_answers,
                # debug=True,
            )
            # tour.visualize()

            with open(
                tmpdir / "out/project/config/everyvoice-shared-text.yaml",
                encoding="utf8",
            ) as f:
                text_config = yaml.safe_load(f)
            self.assertEqual(
                text_config["g2p_engines"],
                {
                    "git": "everyvoice.tests.g2p_engines.g2p_test_upper",
                    "lang1": "everyvoice.tests.g2p_engines.valid",
                },
            )

            # import pprint

            # pprint.pp(tour.state)
            # print(text_config)

            filelist_base = tmpdir / "out/project"
            with open(filelist_base / "dataset0-filelist.psv", encoding="utf8") as f:
                dataset0 = f.read()
            self.assertEqual(
                dataset0,
                dedent(
                    """\
                    basename|language|speaker|characters|phones
                    f1|git|speaker_0|foo bar|FOOBAR
                    f2|git|speaker_0|bar baz|BARBAZ
                    f3|git|speaker_0|baz foo|BAZFOO
                    """
                ),
                "With g2p_engines.valid as custom g2p, phones has spaces stripped",
            )
            with open(filelist_base / "dataset1-filelist.psv", encoding="utf8") as f:
                dataset1 = f.read()
            self.assertIn(
                "\nf7|unknown-lang|my_speaker|zang|\n",
                dataset1,
                "unknown-lang has no g2p engine so phones value is missing",
            )
            self.assertIn(
                "\nf7|lang1|my_speaker|a b c|abc\n",
                dataset1,
                "lang1 uses valid so it strip spaces",
            )
            with open(filelist_base / "dataset2-filelist.psv", encoding="utf8") as f:
                dataset2 = f.read()
            self.assertEqual(
                dataset2,
                dedent(
                    """\
                    basename|language|speaker|characters
                    f1|my-lang|my_speaker|foo bar
                    f2|my-lang|my_speaker|bar baz
                    f3|my-lang|my_speaker|baz foo
                    """
                ),
                "With no g2p engine, the phones column is simply absent",
            )

    def test_custom_g2p_on_second_instance(self):
        """Ensure correct results when you set the custom g2p the second time a language is seen"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with open(tmpdir / "filelist1.psv", "w", encoding="utf8") as f:
                f.write(self.DATASET1)
            with open(tmpdir / "filelist2.psv", "w", encoding="utf8") as f:
                f.write(self.DATASET1)  # reuse DATASET1 (not 2) on purpose
            for i in range(1, 4):
                with open(tmpdir / f"f{i}.wav", "wb"):
                    pass

            second_dataset_children_answers = [
                RecursiveAnswers(
                    patch_questionary(tmpdir / "filelist2.psv")
                ),  # filelist step
                RecursiveAnswers(
                    patch_menu_prompt(1),  # yes to permission
                    children_answers=[
                        RecursiveAnswers(patch_menu_prompt(0)),  # psv format
                        RecursiveAnswers(patch_menu_prompt(0)),  # characters
                        RecursiveAnswers(patch_menu_prompt(())),  # no text prepro
                        RecursiveAnswers(
                            patch_menu_prompt(0),  # no, does not have speaker column
                            children_answers=[
                                RecursiveAnswers(
                                    patch_menu_prompt(1),  # Yes, give speaker ID
                                    children_answers=[
                                        RecursiveAnswers(patch_input("my_speaker"))
                                    ],
                                ),
                            ],
                        ),
                        RecursiveAnswers(
                            patch_menu_prompt(0),  # no, there is no language column
                            children_answers=[RecursiveAnswers(Say("git"))],
                        ),
                        # Keep g2p settings means this project will have one non-g2p-able lang
                        RecursiveAnswers(
                            patch_menu_prompt(1),
                            children_answers=[
                                RecursiveAnswers(
                                    Say("everyvoice.tests.g2p_engines.g2p_test_upper")
                                )
                            ],
                        ),  # Request custom g2p for git
                        RecursiveAnswers(patch_questionary(tmpdir)),  # wav directory
                        RecursiveAnswers(null_patch()),  # ValidateWavsStep
                        RecursiveAnswers(null_patch()),  # SymbolSetStep
                        RecursiveAnswers(patch_menu_prompt([])),  # no Sox
                        RecursiveAnswers(patch_input("dataset1")),  # Dataset name
                    ],
                ),
                RecursiveAnswers(
                    patch_menu_prompt(0),  # no more data
                    children_answers=[RecursiveAnswers(patch_menu_prompt(0))],  # yaml
                ),
            ]

            steps_and_answers = [
                *self.basic_steps,
                StepAndAnswer(
                    basic.OutputPathStep(), patch_questionary(tmpdir / "out")
                ),
                # First dataset
                StepAndAnswer(
                    dataset.FilelistStep(state_subset="dataset_0"),
                    patch_questionary(tmpdir / "filelist1.psv"),
                ),
                StepAndAnswer(
                    dataset.FilelistFormatStep(state_subset="dataset_0"),
                    patch_menu_prompt(0),  # psv
                ),
                StepAndAnswer(
                    dataset.FilelistTextRepresentationStep(state_subset="dataset_0"),
                    patch_menu_prompt(0),  # characters
                ),
                StepAndAnswer(
                    dataset.TextProcessingStep(state_subset="dataset_0"),
                    patch_menu_prompt(()),
                ),
                StepAndAnswer(
                    dataset.HasSpeakerStep(state_subset="dataset_0"),
                    patch_menu_prompt(0),  # 0 is no
                    children_answers=[
                        RecursiveAnswers(
                            patch_menu_prompt(1),  # Yes, give speaker ID
                            children_answers=[
                                RecursiveAnswers(patch_input("my_speaker"))
                            ],
                        ),
                    ],
                ),
                StepAndAnswer(
                    dataset.HasLanguageStep(state_subset="dataset_0"),
                    patch_menu_prompt(0),  # 0 is no
                    children_answers=[RecursiveAnswers(Say("git"))],
                ),
                StepAndAnswer(
                    dataset.CustomG2PStep(state_subset="dataset_0"),
                    patch_menu_prompt(1),  # custom g2p for "git"
                    children_answers=[
                        RecursiveAnswers(Say("everyvoice.tests.g2p_engines.valid")),
                        # RecursiveAnswers( patch_input( ( "everyvoice.tests.g2p_engines.valid",),)),
                        RecursiveAnswers(patch_menu_prompt(0)),  # keep g2p
                    ],
                ),
                StepAndAnswer(
                    dataset.WavsDirStep(state_subset="dataset_0"),
                    patch_questionary(tmpdir),
                ),
                StepAndAnswer(
                    dataset.SymbolSetStep(state_subset="dataset_0"),
                    null_patch(),
                ),
                StepAndAnswer(
                    dataset.SoxEffectsStep(state_subset="dataset_0"),
                    patch_menu_prompt([]),
                ),
                StepAndAnswer(
                    dataset.DatasetNameStep(state_subset="dataset_0"),
                    patch_input("dataset0"),
                ),
                StepAndAnswer(
                    basic.MoreDatasetsStep(),
                    patch_menu_prompt(1),  # 1 is yes
                    children_answers=second_dataset_children_answers,
                ),
            ]

            tour, _ = self.monkey_run_tour(
                "Tour with custom g2p functions overriden between datasets",
                steps_and_answers,
                # debug=True,
            )
            # tour.visualize()

            with open(
                tmpdir / "out/project/config/everyvoice-shared-text.yaml",
                encoding="utf8",
            ) as f:
                text_config = yaml.safe_load(f)
            self.assertEqual(
                text_config["g2p_engines"],
                {"git": "everyvoice.tests.g2p_engines.g2p_test_upper"},
            )

            filelist_base = tmpdir / "out/project"
            with open(filelist_base / "dataset0-filelist.psv", encoding="utf8") as f:
                dataset0 = f.read()
            with open(filelist_base / "dataset1-filelist.psv", encoding="utf8") as f:
                dataset1 = f.read()
            # print(dataset1)
            self.assertEqual(dataset0, dataset1)
            self.assertEqual(
                dataset0,
                dedent(
                    """\
                    basename|language|speaker|characters|phones
                    f1|git|my_speaker|foo bar|FOOBAR
                    f2|git|my_speaker|bar baz|BARBAZ
                    f3|git|my_speaker|baz foo|BAZFOO
                    """
                ),
            )
