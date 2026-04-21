"""
SMILES tokenization (ported from Aromma smilesx_ext_mlx).
SmilesXTokenizer: ext-tools/smilesx style ([], Cl/Br single tokens).
AtomwiseTokenizer: Molecular Transformer style; fewer, larger tokens.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

_ATOMWISE_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|@|\?|>>|\*|\$|%[0-9]{2}|[0-9])"
)

DEFAULT_REPL = {"@@": "_", "@": "^", "Cl": "L", "Br": "R"}
BRACKET = ["[", "]"]
LRB = ["%"]
ALIPHATIC_ORGANIC = ["H", "B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]


def replace(smi: str, repl: Optional[Dict[str, str]] = None) -> str:
    if repl is None:
        repl = DEFAULT_REPL
    for key, val in repl.items():
        smi = smi.replace(key, val)
    return smi


def get_smiles_tokenizer(smiles: str) -> List[str]:
    """Split SMILES into tokens: [], %NN, Cl/Br as single. Matches ext-tools/smilesx."""
    smiles = (smiles or "").replace("\n", "")
    smiles = smiles.replace(BRACKET[0], " " + BRACKET[0]).replace(BRACKET[1], BRACKET[1] + " ")
    lrb_print = [smiles[ic : ic + 3] for ic, _ in enumerate(smiles) if smiles[ic : ic + 1] == LRB[0]]
    for ichar in lrb_print:
        smiles = smiles.replace(ichar, " " + ichar + " ")
    smiles = smiles.split(" ")
    splitted_smiles = []
    for ifrag in smiles:
        if not ifrag:
            continue
        ifrag_tag = any(ac in ifrag for ac in BRACKET + LRB)
        if not ifrag_tag:
            for iaa in ("Cl", "Br"):
                ifrag = ifrag.replace(iaa, " " + iaa + " ")
            ifrag_tmp = ifrag.split(" ")
            for iifrag_tmp in ifrag_tmp:
                if iifrag_tmp not in ("Cl", "Br"):
                    splitted_smiles.extend(list(iifrag_tmp))
                else:
                    splitted_smiles.append(iifrag_tmp)
        else:
            splitted_smiles.append(ifrag)
    return splitted_smiles


def get_smiles_tokenizer_atomwise(smiles: str) -> List[str]:
    """Atom-wise SMILES (Molecular Transformer style). Fewer, larger tokens."""
    smiles = (smiles or "").replace("\n", "").strip()
    if not smiles:
        return []
    return _ATOMWISE_PATTERN.findall(smiles)


class SmilesXTokenizer:
    """Build vocab from molecules via get_smiles_tokenizer; encode to padded indices. pad_id=0, unk_id=1."""

    def __init__(self, maxlen: int = 100):
        self.maxlen = maxlen
        self.tok2int: Dict[str, int] = {"Unk": 1}
        self.int2tok: Dict[int, str] = {1: "Unk"}
        self._pad_id = 0
        self._unk_id = 1

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def unk_id(self) -> int:
        return self._unk_id

    @property
    def vocab_size(self) -> int:
        return max(self.int2tok.keys(), default=1) + 1

    def add_molecules(self, mol_list: List[str]) -> "SmilesXTokenizer":
        for smi in mol_list:
            if not smi or not isinstance(smi, str):
                continue
            toks = get_smiles_tokenizer(smi)
            for t in toks:
                if t not in self.tok2int:
                    idx = max(self.int2tok.keys(), default=1) + 1
                    self.tok2int[t] = idx
                    self.int2tok[idx] = t
        return self

    def encode_one(self, smi: str, pad: bool = True) -> List[int]:
        toks = get_smiles_tokenizer(smi or "")
        ids = [self.tok2int.get(t, self._unk_id) for t in toks[: self.maxlen]]
        if pad and len(ids) < self.maxlen:
            ids += [self._pad_id] * (self.maxlen - len(ids))
        return ids

    def encode_batch(self, smi_list: List[str]) -> np.ndarray:
        rows = [self.encode_one(smi or "", pad=True) for smi in smi_list]
        return np.array(rows, dtype=np.int32)


class AtomwiseTokenizer:
    """Atom-wise tokens; same interface as SmilesXTokenizer."""

    def __init__(self, maxlen: int = 100):
        self.maxlen = maxlen
        self.tok2int: Dict[str, int] = {"Unk": 1}
        self.int2tok: Dict[int, str] = {1: "Unk"}
        self._pad_id = 0
        self._unk_id = 1

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def unk_id(self) -> int:
        return self._unk_id

    @property
    def vocab_size(self) -> int:
        return max(self.int2tok.keys(), default=1) + 1

    def add_molecules(self, mol_list: List[str]) -> "AtomwiseTokenizer":
        for smi in mol_list:
            if not smi or not isinstance(smi, str):
                continue
            toks = get_smiles_tokenizer_atomwise(smi)
            for t in toks:
                if t not in self.tok2int:
                    idx = max(self.int2tok.keys(), default=1) + 1
                    self.tok2int[t] = idx
                    self.int2tok[idx] = t
        return self

    def encode_one(self, smi: str, pad: bool = True) -> List[int]:
        toks = get_smiles_tokenizer_atomwise(smi or "")
        ids = [self.tok2int.get(t, self._unk_id) for t in toks[: self.maxlen]]
        if pad and len(ids) < self.maxlen:
            ids += [self._pad_id] * (self.maxlen - len(ids))
        return ids

    def encode_batch(self, smi_list: List[str]) -> np.ndarray:
        rows = [self.encode_one(smi or "", pad=True) for smi in smi_list]
        return np.array(rows, dtype=np.int32)
