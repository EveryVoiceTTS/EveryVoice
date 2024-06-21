from typing import Dict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from everyvoice.config.shared_types import ConfigModel
from everyvoice.config.utils import PossiblySerializedCallable
from everyvoice.text.utils import normalize_text_helper
from everyvoice.utils import collapse_whitespace


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


class TextConfig(ConfigModel):
    symbols: Symbols = Field(default_factory=Symbols)
    to_replace: Dict[str, str] = {}  # Happens before cleaners
    cleaners: list[PossiblySerializedCallable] = [
        collapse_whitespace,
    ]

    @model_validator(mode="after")
    def clean_symbols(self) -> "TextConfig":
        """We should apply all cleaners to the symbols

        Returns:
            TextConfig: a text config with cleaned symbols
        """
        for k, v in self.symbols:
            if k not in ["punctuation", "silence"]:
                setattr(
                    self.symbols,
                    k,
                    [
                        normalize_text_helper(x, self.to_replace, self.cleaners)
                        for x in v
                    ],
                )
        return self
