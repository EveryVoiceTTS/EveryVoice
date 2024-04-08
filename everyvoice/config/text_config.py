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
        """
        Otherwise we could use the unicodedata categories as in filter out any
        characters with a category that starts with "P".

        * [UNICODE CHARACTER DATABASE](https://unicode.org/reports/tr44/)
        * [General Category Values](https://unicode.org/reports/tr44/#General_Category_Values)
        * [Punctuation and Symbols](https://www.unicode.org/faq/punctuation_symbols.html)
        * [Unicode Properties](https://docs.python.org/3/howto/unicode.html#unicode-properties)

        In [1]: import unicodedata
        In [2]: { c: unicodedata.category(c) for c in {'"', '—', '.', '?', '¿', '“', "'", ';', '«', '…', '»', '¡', '”', '!', ':', '-', ',', "a", "2"}}
        Out[2]:
        {'¡': 'Po',
        ',': 'Po',
        '”': 'Pf',
        '.': 'Po',
        'a': 'Ll',
        ';': 'Po',
        '¿': 'Po',
        "'": 'Po',
        ':': 'Po',
        '«': 'Pi',
        '"': 'Po',
        '-': 'Pd',
        '“': 'Pi',
        '!': 'Po',
        '…': 'Po',
        '2': 'Nd',
        '?': 'Po',
        '—': 'Pd',
        '»': 'Pf'}
        """
        dataset_names = filter(
            lambda dn: dn.endswith("_characters"),
            dict(self.model_dump()).keys(),
        )
        punctuation = self.punctuation.all | set(" ")
        for dataset_name in dataset_names:
            setattr(
                self, dataset_name, list(set(getattr(self, dataset_name)) - punctuation)
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
