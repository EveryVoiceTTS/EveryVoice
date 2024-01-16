from typing import Dict, List, Union

from pydantic import BaseModel, ConfigDict, Field

from everyvoice.config.shared_types import ConfigModel
from everyvoice.config.utils import PossiblySerializedCallable
from everyvoice.utils import collapse_whitespace, lower, nfc_normalize


class Symbols(BaseModel):
    silence: Union[str, List[str]] = Field(
        ["<SIL>"], description="The symbol(s) used to indicate silence."
    )
    pad: str = Field(
        "_",
        description="The symbol used to indicate padding. Batches are length-normalized by adding this padding character so that each utterance in the batch is the same length.",
    )
    punctuation: Union[str, List[str]] = Field(
        "-';:,.!?¡¿—…\"«»“” ", description="A list of punctuation symbols."
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
