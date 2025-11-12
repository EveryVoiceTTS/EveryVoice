import importlib
from pathlib import Path
from typing import Annotated

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from everyvoice.config.shared_types import ConfigModel, init_context
from everyvoice.config.utils import PossiblySerializedCallable
from everyvoice.text.phonemizer import G2PCallable
from everyvoice.text.utils import normalize_text_helper
from everyvoice.utils import (
    collapse_whitespace,
    load_config_from_json_or_yaml_path,
    strip_text,
)


class Punctuation(BaseModel):
    exclamations: list[str] = Field(
        # TODO: consider how to handle utt final punctuation like ! ? and .
        default=["!", "¡"],
        description="Exclamation punctuation symbols used in your datasets. Replaces these symbols with <EXCL> internally.",
    )
    question_symbols: list[str] = Field(
        default=["?", "¿"],
        description="Question/interrogative punctuation symbols used in your datasets. Replaces these symbols with <QINT> internally.",
    )
    quotemarks: list[str] = Field(
        default=['"', "'", "“", "”", "«", "»"],
        description="Quotemark punctuation symbols used in your datasets. Replaces these symbols with <QUOTE> internally.",
    )
    parentheses: list[str] = Field(
        default=["(", ")", "[", "]", "{", "}"],
        description="Punctuation symbols indicating parentheses, brackets, or braces. Replaces these symbols with <PAREN> internally.",
    )
    periods: list[str] = Field(
        default=["."],
        description="Punctuation symbols indicating a 'period' used in your datasets. Replaces these symbols with <PERIOD> internally.",
    )
    colons: list[str] = Field(
        default=[":"],
        description="Punctuation symbols indicating a 'colon' used in your datasets. Replaces these symbols with <COLON> internally.",
    )
    semi_colons: list[str] = Field(
        default=[";"],
        description="Punctuation symbols indicating a 'semi-colon' used in your datasets. Replaces these symbols with <SEMICOL> internally.",
    )
    hyphens: list[str] = Field(
        default=["-", "—", "*"],
        description="Punctuation symbols indicating a 'hyphen' used in your datasets. * is a hyphen by default since unidecode decodes middle-dot punctuation as an asterisk. Replaces these symbols with <HYPHEN> internally.",
    )
    commas: list[str] = Field(
        default=[","],
        description="Punctuation symbols indicating a 'comma' used in your datasets. Replaces these symbols with <COMMA> internally.",
    )
    ellipses: list[str] = Field(
        default=["…"],
        description="Punctuation symbols indicating ellipses used in your datasets. Replaces these symbols with <EPS> internally.",
    )

    @property
    def all(self) -> set[str]:
        """Return a set of all punctuations."""
        return (
            set(self.exclamations)
            | set(self.question_symbols)
            | set(self.quotemarks)
            | set(self.periods)
            | set(self.colons)
            | set(self.semi_colons)
            | set(self.hyphens)
            | set(self.commas)
            | set(self.parentheses)
            | set(self.ellipses)
        )


class Symbols(BaseModel):
    silence: list[str] = Field(
        default=["<SIL>"], description="The symbol(s) used to indicate silence."
    )
    punctuation: Punctuation = Field(
        default_factory=Punctuation,
        description="EveryVoice will combine punctuation and normalize it into a set of five permissible types of punctuation to help tractable training.",
    )
    model_config = ConfigDict(extra="allow")

    @property
    def all_except_punctuation(self) -> set[str]:
        """Returns the set containing all characters."""
        return set(w for _, v in self if not isinstance(v, Punctuation) for w in v)

    @model_validator(mode="after")
    def cannot_have_punctuation_in_symbol_set(self) -> "Symbols":
        """You cannot have the same symbol defined in punctuation as elsewhere.

        Raises:
            ValueError: raised if a symbol from punctuation is found elsewhere

        Returns:
            Symbols: The validated symbol set
        """
        for punctuation in self.punctuation.all:
            if punctuation in self.all_except_punctuation:
                raise ValueError(
                    f"Sorry, the symbol '{punctuation}' occurs in both your declared punctuation and in your other symbol set. Please inspect your text configuration and either remove the symbol from the punctuation or other symbol set."
                )
        return self

    @model_validator(mode="after")
    def member_must_be_list_of_strings(self) -> "Symbols":
        """Except for `punctuation` & `pad`, all user defined member variables
        have to be a list of strings.
        """
        for k, v in self:
            if isinstance(v, Punctuation):
                continue
            if k == "pad":
                continue
            if not isinstance(v, list) or not all(isinstance(e, str) for e in v):
                raise ValueError(f"{k} must be a list")

        return self


def get_label_from_symbol_key(key: str) -> str | None:
    """Given a symbol key like dataset1_phones or punctuation, return the dataset label
    if key matches *_phones or *_characters where * is the label, or else None"""
    last_underscore = key.rfind("_")
    if last_underscore >= 1 and key[last_underscore + 1 :] in (
        "phones",
        "characters",
    ):
        return key[:last_underscore]
    else:
        return None


Language = Annotated[
    str,
    Field(
        title="Language ID",
        examples=["fr"],
    ),
]
G2P_py_module = Annotated[
    str,
    Field(
        title="Module path",
        examples=["everyvoice_plugin_g2p4example.g2p"],
    ),
]
G2P_Engines = Annotated[
    dict[Language, G2P_py_module],
    Field(description="Mapping from language id to g2p module"),
]


class LanguageBoundaries(BaseModel):
    strong: str = Field(
        default="!?.",
        description="All characters that constitute strong boundaries, for one language.",
    )
    weak: str = Field(
        default=":;,",
        description="All characters that constitute strong boundaries, for one language.",
    )


def validate_g2p_engine_signature(g2p_func: G2PCallable) -> G2PCallable:
    """
    A G2P engine's signature should be:

    Callable[[str], List[str]]

    Note that we have to use `List` and not `list`.
    """
    import typing
    from inspect import signature

    sig = signature(g2p_func)
    assert (
        len(sig.parameters) == 1
    ), "G2P engine's signature should take a single argument"
    arg_names = list(sig.parameters)
    assert (
        sig.parameters[arg_names[0]].annotation is str
    ), "G2P Engine's signature should take a string"
    assert (
        sig.return_annotation is typing.List[str]
    ), "G2P Engine's signature should return a list of strings"

    return g2p_func


def load_custom_g2p_engine(lang_id: str, qualified_g2p_func_name: str) -> G2PCallable:
    # Load the user provided G2P Engine.
    module_name, _, function_name = qualified_g2p_func_name.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        error_message = f"Invalid G2P engine module `{module_name}` for `{lang_id}`"
        logger.error(error_message)
        raise ValueError(error_message)

    return validate_g2p_engine_signature(getattr(module, function_name))


DEFAULT_CLEANERS: list[PossiblySerializedCallable] = [collapse_whitespace, strip_text]


class TextConfig(ConfigModel):
    symbols: Symbols = Field(default_factory=Symbols)
    to_replace: dict[str, str] = Field(
        default={},
        title="Global text replacements",
        description="Map of match-to-replacement to apply on training and run-time text, before cleaners are applied. Superceded by language_to_replace when processing text in a language which has language-specific text replacements, which are in turn superceded by dataset_to_replace when processing a dataset which has dataset-specific text replacements.",
    )
    language_to_replace: dict[str, dict[str, str]] = Field(
        default={},
        title="Language-specific text replacements",
        description="Map from language code to text replacement maps. Supercedes the global text replacements when defined for a given language. Superceded by dataset_to_replace when processing a dataset which has dataset-specific text replacements.",
    )
    dataset_to_replace: dict[str, dict[str, str]] = Field(
        default={},
        title="Dataset-specific text replacements.",
        description="Map from dataset label to replacement maps. Supercedes both the global text replacements and language_to_replace when defined for a given dataset.",
    )
    cleaners: list[PossiblySerializedCallable] = Field(
        default=DEFAULT_CLEANERS,
        title="Global cleaners",
        description="List of cleaners to apply to all datasets and run-time data. Superceded by language_cleaners when processing text in a language which has language-specific cleaners, which are in turn superceded by dataset_cleaners when processing a dataset which has dataset-specific cleaners.",
    )
    language_cleaners: dict[str, list[PossiblySerializedCallable]] = Field(
        default={},
        title="Language-specific cleaners",
        description="Map from language code to cleaner lists. Supercedes the global cleaners when defined for a given language. Superceded by dataset_cleaners when processing a dataset which has dataset-specific cleaners.",
    )
    dataset_cleaners: dict[str, list[PossiblySerializedCallable]] = Field(
        default={},
        title="Dataset-specific cleaners",
        description="Map from dataset label to cleaner lists. Supercedes both the global cleaners and language_cleaners when defined for a given dataset.",
    )
    g2p_engines: G2P_Engines = Field(
        default={},
        title="External G2P",
        description="User defined or external G2P engines.\nSee https://github.com/EveryVoiceTTS/everyvoice_g2p_template_plugin to implement your own G2P.",
        examples=["""{"fr": "everyvoice_plugin_g2p4example.g2p"}"""],
    )
    split_text: bool = Field(
        default=True,
        title="Split Text",
        description="Whether or not to perform text splitting (also referred to as text chunking) at inference time. Instead of synthesizing an entire utterance, the utterance will be split into smaller chunks and re-combined after synthesis. This can lead to more natural synthesis for long-form (i.e. paragraph) synthesis.",
    )
    boundaries: dict[Language, LanguageBoundaries] = Field(
        default={},
        title="Boundaries",
        description="Strong and Weak boundaries on which text splitting is to be performed, for every language.",
        examples=["""{'eng': {'strong': '!?.', 'weak': ':;,'}}'"""],
    )

    def get_cleaners(
        self, *, lang_id: str | None = None, dataset_label: str | None = None
    ) -> list[PossiblySerializedCallable]:
        """Get the cleaners to apply to a given dataset and language

        Dataset has top precendence, then language, falling back to global cleaners
        """
        if dataset_label is not None and dataset_label in self.dataset_cleaners:
            return self.dataset_cleaners[dataset_label]
        elif lang_id is not None and lang_id in self.language_cleaners:
            return self.language_cleaners[lang_id]
        else:
            return self.cleaners

    def get_to_replace(
        self, *, lang_id: str | None = None, dataset_label: str | None = None
    ) -> dict[str, str]:
        """Get the to_replace filters to apply to a given dataset and language

        Dataset has top precendence, then language, falling back to global cleaners
        """
        if dataset_label is not None and dataset_label in self.dataset_to_replace:
            return self.dataset_to_replace[dataset_label]
        elif lang_id is not None and lang_id in self.language_to_replace:
            return self.language_to_replace[lang_id]
        else:
            return self.to_replace

    @model_validator(mode="after")
    def clean_symbols(self) -> Self:
        """We should apply all cleaners to the symbols

        Returns:
            TextConfig: a text config with cleaned symbols
        """
        for k, v in self.symbols:
            if k not in ["punctuation", "silence"]:
                dataset_label = get_label_from_symbol_key(k)
                cleaners = self.get_cleaners(dataset_label=dataset_label)
                to_replace = self.get_to_replace(dataset_label=dataset_label)
                normalized = [normalize_text_helper(x, to_replace, cleaners) for x in v]
                setattr(self.symbols, k, normalized)

                if "" in normalized or len(normalized) != len(set(normalized)):
                    logger.warning(
                        f"Normalization created a duplicate or inserted '' in {k}={normalized}. "
                        "Please check your shared-text config for problems."
                    )

        return self

    @model_validator(mode="after")
    def load_g2p_engines(self) -> Self:
        """
        Given `g2p_engines`, populate the global list `AVAILABLE_G2P_ENGINES`.
        """
        from everyvoice.text.phonemizer import AVAILABLE_G2P_ENGINES

        for lang_id, name in self.g2p_engines.items():
            g2p_func = load_custom_g2p_engine(lang_id, name)

            if lang_id in AVAILABLE_G2P_ENGINES:
                logger.warning(
                    f"Overriding g2p for `{lang_id}` with user provided g2p plugin `{name}`"
                )

            AVAILABLE_G2P_ENGINES[lang_id] = g2p_func
            logger.info(f"Adding G2P engine from `{name}` for `{lang_id}`")

        return self

    @staticmethod
    def load_config_from_path(path: Path) -> "TextConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        with init_context({"config_path": path}):
            config = TextConfig(**config)
        return config
