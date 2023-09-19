from typing import Dict, List, Union

from pydantic import BaseModel, ConfigDict, Field

from everyvoice.config.shared_types import ConfigModel
from everyvoice.config.utils import PossiblySerializedCallable
from everyvoice.utils import collapse_whitespace, lower, nfc_normalize


class Symbols(BaseModel):
    silence: Union[str, List[str]] = ["<SIL>"]
    pad: str = "_"
    punctuation: Union[str, List[str]] = "-';:,.!?¡¿—…\"«»“” "
    model_config = ConfigDict(extra="allow")


class TextConfig(ConfigModel):
    symbols: Symbols = Field(default_factory=Symbols)
    to_replace: Dict[str, str] = {}  # Happens before cleaners
    cleaners: List[PossiblySerializedCallable] = [
        lower,
        collapse_whitespace,
        nfc_normalize,
    ]
