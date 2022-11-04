from typing import Callable, List, Union

from pydantic import BaseModel, Extra

from smts.config.shared_types import ConfigModel
from smts.config.utils import string_to_callable


class Symbols(BaseModel):
    silence: Union[str, List[str]]
    pad: str
    punctuation: Union[str, List[str]]

    class Config:
        extra = Extra.allow


class TextConfig(ConfigModel):
    symbols: Symbols
    cleaners: List[Callable]

    def __init__(self, cleaners: List[Union[str, Callable]], **data) -> None:
        """Custom init to process cleaners"""
        for i, c in enumerate(cleaners):
            cleaners[i] = string_to_callable(c)
        super().__init__(cleaners=cleaners, **data)
