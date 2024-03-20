from typing import Dict, List, Union

from pydantic import BaseModel, ConfigDict, Field

from everyvoice.config.shared_types import ConfigModel
from everyvoice.config.utils import PossiblySerializedCallable
from everyvoice.utils import collapse_whitespace, lower, nfc_normalize


class Punctuation(BaseModel):
    exclamations: List[str] = Field(
        ["!", "¡"],
        description="Exclamation punctuation symbols used in your datasets. Replaces these symbols with <EXCL> internally.",
    )
    question_symbols: List[str] = Field(
        ["?", "¿"],
        description="Question/interrogative punctuation symbols used in your datasets. Replaces these symbols with <QINT> internally.",
    )
    quotemarks: List[str] = Field(
        ['"', "'", "“", "”", "«", "»"],
        description="Quotemark punctuation symbols used in your datasets. Replaces these symbols with <QUOTE> internally.",
    )
    big_breaks: List[str] = Field(
        [".", ":", ";", "…"],
        description="Punctuation symbols indicating a 'big break' used in your datasets. Replaces these symbols with <BB> internally.",
    )
    small_breaks: List[str] = Field(
        [",", "-", "—"],
        description="Punctuation symbols indicating a 'small break' used in your datasets. Replaces these symbols with <SB> internally.",
    )


class Symbols(BaseModel):
    silence: Union[str, List[str]] = Field(
        ["<SIL>"], description="The symbol(s) used to indicate silence."
    )
    pad: str = Field(
        "_",
        description="The symbol used to indicate padding. Batches are length-normalized by adding this padding character so that each utterance in the batch is the same length.",
    )
    punctuation: Punctuation = Field(
        default_factory=Punctuation,
        description="EveryVoice will combine punctuation and normalize it into a set of five permissible types of punctuation to help tractable training.",
    )
    model_config = ConfigDict(extra="allow")


class TextConfig(ConfigModel):
    symbols: Symbols = Field(default_factory=Symbols)
    to_replace: Dict[str, str] = {}  # Happens before cleaners
    cleaners: List[PossiblySerializedCallable] = [
        lower,
        collapse_whitespace,
        nfc_normalize,
    ]
