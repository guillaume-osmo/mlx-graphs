"""
Group-contribution based atom annotations for GroupGAT/AGC models.

Uses first-order groups from Hukkerikar et al. (MG_plus_reference, GC-GNN repo).
Reference: Aouichaoui et al., JCIM 2023, https://doi.org/10.1021/acs.jcim.2c01091
Repository: https://github.com/gsi-lab/GC-GNN
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from rdkit import Chem

# Default path: same dir as this module
_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_CSV = _MODULE_DIR / "MG_plus_reference.csv"

# Fallback: try upstream URL for first load (user can cache)
_GC_GNN_CSV_URL = "https://raw.githubusercontent.com/gsi-lab/GC-GNN/main/datasets/MG_plus_reference.csv"


def _load_mg_reference(path: str | Path | None = None) -> list[tuple[str, int, str]]:
    """Load MG_plus_reference: (group_name, priority, smarts) sorted by priority ascending."""
    p = path or _DEFAULT_CSV
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"MG_plus_reference.csv not found at {p}. "
            f"Download from {_GC_GNN_CSV_URL} and place in mlx_graphs/chem/"
        )
    rows = []
    with open(p, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for r in reader:
            group = (r.get("First-Order Group", "") or r.get("First-Order Group", "")).strip()
            if not group:
                continue
            try:
                prio = int(r.get("Priority", 17))
            except (ValueError, TypeError):
                prio = 17
            smarts = (r.get("SMARTs", "") or "").strip()
            rows.append((group, prio, smarts))
    # Sort by priority ascending (lower = higher priority)
    rows.sort(key=lambda x: x[1])
    return rows


def _ensure_mg_reference() -> Path:
    """Return path to MG_plus_reference.csv, downloading if missing."""
    if _DEFAULT_CSV.exists():
        return _DEFAULT_CSV
    try:
        import urllib.request

        urllib.request.urlretrieve(_GC_GNN_CSV_URL, str(_DEFAULT_CSV))
        return _DEFAULT_CSV
    except Exception:
        raise FileNotFoundError(
            f"MG_plus_reference.csv not found. Download from {_GC_GNN_CSV_URL} "
            f"and save to {_DEFAULT_CSV}"
        )


class GroupContributionFeaturizer:
    """Assign atoms to first-order groups (Hukkerikar) for GroupGAT-style augmentation."""

    def __init__(self, csv_path: str | Path | None = None):
        try:
            path = csv_path or _ensure_mg_reference()
        except FileNotFoundError:
            path = csv_path or _DEFAULT_CSV
        self.rows = _load_mg_reference(path)
        self.group_names = [r[0] for r in self.rows]
        self.num_groups = len(self.group_names)
        self._smarts_cache: list | None = None

    @property
    def smarts_patterns(self) -> list:
        """Lazy compile SMARTS."""
        if self._smarts_cache is None:
            self._smarts_cache = []
            for _, _, sm in self.rows:
                try:
                    pat = Chem.MolFromSmarts(sm)
                    self._smarts_cache.append(pat)
                except Exception:
                    self._smarts_cache.append(None)
        return self._smarts_cache

    def mol_to_group_ids(self, mol) -> np.ndarray:
        """Assign each atom to one group (priority-ordered). Returns shape (num_atoms,) int."""
        num_atoms = mol.GetNumAtoms()
        atom_to_group = np.full(num_atoms, -1, dtype=np.int32)
        assigned = set()

        for gid, (_, _, _) in enumerate(self.rows):
            pat = self.smarts_patterns[gid]
            if pat is None:
                continue
            try:
                matches = mol.GetSubstructMatches(pat, uniquify=True)
            except Exception:
                continue
            for match in matches:
                for aid in match:
                    if aid not in assigned:
                        atom_to_group[aid] = gid
                        assigned.add(aid)

        # Unassigned atoms -> last group (fallback)
        atom_to_group[atom_to_group < 0] = self.num_groups - 1
        return atom_to_group

    def mol_to_group_onehot(self, mol) -> np.ndarray:
        """One-hot group membership per atom. Shape (num_atoms, num_groups)."""
        ids = self.mol_to_group_ids(mol)
        n, k = len(ids), self.num_groups
        oh = np.zeros((n, k), dtype=np.float32)
        oh[np.arange(n), np.clip(ids, 0, k - 1)] = 1.0
        return oh


def augment_node_features_with_groups(
    node_features: np.ndarray,
    mol,
    featurizer: GroupContributionFeaturizer | None = None,
) -> np.ndarray:
    """Concatenate group one-hot to node features. Returns (num_atoms, n_atom + n_groups)."""
    if featurizer is None:
        featurizer = GroupContributionFeaturizer()
    group_oh = featurizer.mol_to_group_onehot(mol)
    return np.concatenate([node_features, group_oh], axis=-1)
