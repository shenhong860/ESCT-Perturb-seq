"""Pure-Python subset of torchtext.vocab used by scgpt."""

from collections import OrderedDict
from typing import Dict, Iterable, List, Optional


class Vocab:
    def __init__(self, vocab: OrderedDict):
        self.vocab = OrderedDict()
        self._stoi: Dict[str, int] = {}
        self._itos: Dict[int, str] = {}
        self._default_index: Optional[int] = None
        for token, index in vocab.items():
            self.insert_token(token, index)

    def __len__(self) -> int:
        return len(self._stoi)

    def __getitem__(self, token: str) -> int:
        if token in self._stoi:
            return self._stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(token)

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def get_stoi(self) -> Dict[str, int]:
        return dict(self._stoi)

    def get_itos(self) -> List[str]:
        return [self._itos[i] for i in sorted(self._itos)]

    def insert_token(self, token: str, index: int) -> None:
        if index < 0:
            raise RuntimeError("token index cannot be negative")
        self._stoi[token] = index
        self._itos[index] = token
        self.vocab[token] = index

    def append_token(self, token: str) -> None:
        self.insert_token(token, len(self._stoi))

    def set_default_index(self, index: Optional[int]) -> None:
        self._default_index = index

    def set_default_token(self, token: str) -> None:
        if token not in self._stoi:
            raise RuntimeError(f"{token} is not in the vocabulary")
        self._default_index = self._stoi[token]


def vocab(ordered_dict: OrderedDict, min_freq: int = 1) -> Vocab:
    return Vocab(ordered_dict)


def build_vocab_from_iterator(
    iterator: Iterable,
    min_freq: int = 1,
    specials: Optional[List[str]] = None,
    special_first: bool = True,
) -> Vocab:
    counter = OrderedDict()
    for tokens in iterator:
        for tok in tokens:
            counter[tok] = counter.get(tok, 0) + 1
    out = OrderedDict()
    if specials:
        sp = list(reversed(specials)) if special_first else list(specials)
        for s in sp:
            out[s] = len(out)
    for tok, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        if freq >= min_freq and tok not in out:
            out[tok] = len(out)
    return Vocab(out)
