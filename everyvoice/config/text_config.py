from typing import Annotated, Dict

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from everyvoice.config.shared_types import ConfigModel
from everyvoice.config.utils import PossiblySerializedCallable
from everyvoice.text.phonemizer import G2PCallable
from everyvoice.text.utils import normalize_text_helper
from everyvoice.utils import collapse_whitespace, strip_text


class Punctuation(BaseModel):
    exclamations: list[str] = Field(
        ["!", "¡"],  # TODO: consider how to handle utt final punctuation like ! ? and .
        description="Exclamation punctuation symbols used in your datasets. Replaces these symbols with <EXCL> internally.",
    )
    question_symbols: list[str] = Field(
        ["?", "¿"],
        description="Question/interrogative punctuation symbols used in your datasets. Replaces these symbols with <QINT> internally.",
    )
    quotemarks: list[str] = Field(
        ['"', "'", "“", "”", "«", "»"],
        description="Quotemark punctuation symbols used in your datasets. Replaces these symbols with <QUOTE> internally.",
    )
    big_breaks: list[str] = Field(
        [".", ":", ";"],
        description="Punctuation symbols indicating a 'big break' used in your datasets. Replaces these symbols with <BB> internally.",
    )
    small_breaks: list[str] = Field(
        [",", "-", "—"],
        description="Punctuation symbols indicating a 'small break' used in your datasets. Replaces these symbols with <SB> internally.",
    )
    ellipsis: list[str] = Field(
        ["…"],
        description="Punctuation symbols indicating an ellipsis used in your datasets. Replaces these symbols with <EPS> internally.",
    )

    @property
    def all(self) -> set[str]:
        """Return a set of all punctuations."""
        return (
            set(self.exclamations)
            | set(self.question_symbols)
            | set(self.quotemarks)
            | set(self.big_breaks)
            | set(self.small_breaks)
            | set(self.ellipsis)
        )


class Symbols(BaseModel):
    silence: list[str] = Field(
        ["<SIL>"], description="The symbol(s) used to indicate silence."
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


G2P_lang = Annotated[
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
    dict[G2P_lang, G2P_py_module],
    Field(description="Mapping from language id to g2p module"),
]


def _validate_g2p_engine_signature(g2p_func: G2PCallable) -> G2PCallable:
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


class TextConfig(ConfigModel):
    symbols: Symbols = Field(default_factory=Symbols)
    to_replace: Dict[str, str] = {}  # Happens before cleaners
    cleaners: list[PossiblySerializedCallable] = [collapse_whitespace, strip_text]
    g2p_engines: G2P_Engines = Field(
        {},
        title="External G2P",
        description="User defined or external G2P engines.\nSee https://github.com/EveryVoiceTTS/everyvoice_g2p_template_plugin to implement your own G2P.",
        examples=["""{"fr": "everyvoice_plugin_g2p4example.g2p"}"""],
    )

    @model_validator(mode="after")
    def clean_symbols(self) -> Self:
        """We should apply all cleaners to the symbols

        Returns:
            TextConfig: a text config with cleaned symbols
        """
        for k, v in self.symbols:
            if k not in ["punctuation", "silence"]:
                normalized = [
                    normalize_text_helper(x, self.to_replace, self.cleaners) for x in v
                ]
                if "" in normalized or len(normalized) != len(set(normalized)):
                    logger.warning(
                        f"Normalization created a duplicate or inserted '' in {k}={normalized}. "
                        "Please check your shared-text config for problems."
                    )
                setattr(self.symbols, k, normalized)
        return self

    @model_validator(mode="after")
    def load_g2p_engines(self) -> Self:
        """
        Given `g2p_engines`, populate the global list `AVAILABLE_G2P_ENGINES`.
        """
        import importlib

        from everyvoice.text.phonemizer import AVAILABLE_G2P_ENGINES

        for lang_id, name in self.g2p_engines.items():
            # Load the user provided G2P Engine.
            try:
                module_name, _, function_name = name.rpartition(".")
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                error_message = (
                    f"Invalid G2P engine module `{module_name}` for `{lang_id}`"
                )
                logger.error(error_message)
                raise ValueError(error_message)

            g2p_func = _validate_g2p_engine_signature(getattr(module, function_name))

            if lang_id in AVAILABLE_G2P_ENGINES:
                logger.warning(
                    f"Overriding g2p for `{lang_id}` with user provided g2p plugin `{name}`"
                )

            AVAILABLE_G2P_ENGINES[lang_id] = g2p_func
            logger.info(f"Adding G2P engine from `{name}` for `{lang_id}`")

        return self
