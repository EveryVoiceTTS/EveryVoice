import contextlib
from typing import Callable, Dict, List, Union

from pydantic import BaseModel, Extra, Field

from everyvoice.config.shared_types import ConfigModel
from everyvoice.config.utils import string_to_callable
from everyvoice.utils import collapse_whitespace, lower, nfc_normalize


class Symbols(BaseModel):
    silence: Union[str, List[str]] = ["<SIL>"]
    pad: str = "_"
    punctuation: Union[str, List[str]] = "-';:,.!?¡¿—…\"«»“” "

    class Config:
        extra = Extra.allow


class TextConfig(ConfigModel):
    symbols: Symbols = Field(default_factory=Symbols)
    to_replace: Dict[str, str] = {}  # Happens before cleaners
    cleaners: List[Callable] = [lower, collapse_whitespace, nfc_normalize]

    def __init__(self, **data) -> None:
        """Custom init to process cleaners"""
        with contextlib.suppress(KeyError):
            cleaners = data["cleaners"]
            for i, c in enumerate(cleaners):
                cleaners[i] = string_to_callable(c)
        super().__init__(**data)
