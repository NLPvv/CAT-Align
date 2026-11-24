"""Read the pronunciation lexicons published with Whistle checkpoints."""

import pickle
import re
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence


WORD_PATTERN = re.compile(r"[^\W_]+(?:['’ʼ_][^\W_]+)*")
APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'"})


class LexiconState:
    def __setstate__(self, state):
        self.state = state


class LexiconUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if name == "LexiconTokenizer" and module in (
            "cat.shared.tokenizer",
            "ctc_model.tokenizer",
        ):
            return LexiconState
        raise pickle.UnpicklingError(f"Unexpected tokenizer reference {module}.{name}")


def word_key(word: str) -> str:
    return unicodedata.normalize("NFC", word.translate(APOSTROPHES))


class Tokenizer:
    def __init__(
        self, words: Mapping[str, Sequence[int]], units: Mapping[str, int], bos: str, unk: str
    ) -> None:
        self.word_ids = words
        self.unit_ids = units
        self.tokens = {index: unit for unit, index in units.items()}
        self.special_ids = {0, units[bos], units[unk]}

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        with Path(path).open("rb") as file:
            lexicon = LexiconUnpickler(file).load()

        state = getattr(lexicon, "state", None)
        # CTC-TTS uses the short keys and older CTC-TTS/CAT store attributes.
        words = state.get("w2pid", state.get("_w2pid"))
        units = state.get("units", state.get("_units"))
        bos = state.get("bos", state.get("_bos_token"))
        unk = state.get("unk", state.get("_unk_token"))
        
        return cls(words, units, bos, unk)

    def encode_words(
        self, text: str, pronunciations: Mapping[str, str] | None = None
    ) -> list[tuple[str, list[int]]]:
        overrides = {
            word_key(word).lower(): phones for word, phones in (pronunciations or {}).items()
        }
        words = []
        for word in WORD_PATTERN.findall(unicodedata.normalize("NFC", text)):
            key = word_key(word)
            lower = key.lower()
            if lower in overrides:
                ids = self.encode_phones(overrides[lower])
            else:
                ids = self.word_ids.get(key, self.word_ids.get(lower))
                if ids is None:
                    raise ValueError(
                        f"Word {word!r} is missing from the model lexicon. "
                        "Pass its whitespace-separated phonemes in pronunciations."
                    )
                ids = self.check_ids(ids, word)
            words.append((word, ids))

        if not words:
            raise ValueError("Transcript must contain at least one word")
        return words

    def encode_phones(self, phones: str | Sequence[str]) -> list[int]:
        if isinstance(phones, str):
            phones = phones.split()
        try:
            ids = [self.unit_ids[phone] for phone in phones]
        except KeyError as error:
            raise ValueError(
                f"Phoneme {error.args[0]!r} is missing from the model vocabulary"
            ) from error
        return self.check_ids(ids, "phoneme sequence")

    def check_ids(self, ids: Sequence[int], label: str) -> list[int]:
        if not isinstance(ids, (list, tuple)) or not ids or any(
            type(index) is not int or index not in self.tokens for index in ids
        ):
            raise ValueError(f"Invalid token IDs for {label!r}")
        if any(index in self.special_ids for index in ids):
            raise ValueError(f"Pronunciation for {label!r} contains a blank or unknown token")
        return list(ids)
