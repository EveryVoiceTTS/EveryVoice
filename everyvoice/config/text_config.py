from typing import Dict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from everyvoice.config.shared_types import ConfigModel
from everyvoice.config.utils import PossiblySerializedCallable
from everyvoice.utils import collapse_whitespace, lower, nfc_normalize


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
    def all(self):
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

    @model_validator(mode="after")
    def no_punctuation(self) -> "Symbols":
        """
        Ensure that there aren't any characters that are defined in the
        punctuation set that exist in other character lists.
        """
        punctuation = self.punctuation.all | set(" ")
        needs_cleanup_from_user = []
        for dataset_name, symbols in dict(self.model_dump()).items():
            common_symbols = set(symbols) & punctuation
            if common_symbols:
                needs_cleanup_from_user.append(
                    f"Dataset {dataset_name} needs attention for {common_symbols}"
                )

        if needs_cleanup_from_user:
            raise ValueError(
                "Your symbols for the following dataset(s) require(s) some user attention.\n",
                "\n".join(needs_cleanup_from_user),
            )

        return self


class TextConfig(ConfigModel):
    symbols: Symbols = Field(default_factory=Symbols)
    to_replace: Dict[str, str] = {}  # Happens before cleaners
    cleaners: list[PossiblySerializedCallable] = [
        lower,
        collapse_whitespace,
        nfc_normalize,
    ]
