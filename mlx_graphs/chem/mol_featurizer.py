"""
Central molecular featurizer: single source of truth for atom and bond features.

Modes:
- default: RDKit GetAtomFeatures (49) + stereo/btype/conj/rot (4) - DMPNN/ChemProp compatible
- rigr: RIGR resonance-invariant (66 atom, 1 bond)
- merged: Union of unique semantics - GroupGAT-aligned where possible.
  Atom: atomic_num + degree + num_h + mass + formal_charge + explicit_valency (GroupGAT) +
        hybridization + aromatic + chiral
  Bond: bond_stereo_6hot (GroupGAT) + bond_type + conjugated + in_ring + rotatable

Reference: GroupGAT paper (Aouichaoui et al., JCIM 2023)
https://doi.org/10.1021/acs.jcim.2c01091
Tables 1 & 2: atom type, num_bonds, num_h, explicit_valency, hybridization, aromaticity,
chirality_center, chirality_type, formal_charge; bond_type, conjugation, ring, bond_stereo.

All callers (load_esol, compare_esol, feature_selection) import from here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolDescriptors

FeaturizerMode = Literal["default", "rigr", "merged"]

# --- Dimensions ---
N_ATOM = 49
N_BOND = 4
RIGR_ATOM_DIM = 66
RIGR_BOND_DIM = 1

# Merged: semantic composition + GroupGAT extras (explicit_valency, bond_stereo 6-d)
# Atom: atomic_num(54) + degree(6) + num_h(5) + mass(1) + formal_charge(5) + explicit_valency(6) +
#       hybrid(7) + aromatic(1) + chiral(4)
MERGED_ATOM_DIM = 54 + 6 + 5 + 1 + 5 + 6 + 7 + 1 + 4
# Bond: bond_stereo_onehot(6) + bond_type(1) + conjugated(1) + in_ring(1) + rotatable(1)
MERGED_BOND_DIM = 6 + 1 + 1 + 1 + 1  # 10

# --- Feature names for GNN atom/bond (for interpretability, logs, exports) ---
# GroupGAT reference (Aouichaoui et al., JCIM 2023)
GROUPGAT_ATOM_NAMES = [
    ("atom_type", 9),
    ("num_bonds", 5),
    ("num_h", 5),
    ("explicit_valency", 6),
    ("hybridization", 5),
    ("aromaticity", 1),
    ("chirality_center", 1),
    ("chirality_type", 2),
    ("formal_charge", 1),
]
GROUPGAT_BOND_NAMES = [
    ("bond_type", 4),
    ("conjugation", 1),
    ("ring", 1),
    ("bond_stereo", 6),
]

# Merged mode (semantic blocks with sizes)
MERGED_ATOM_NAMES = [
    ("atomic_num", 54),
    ("degree", 6),
    ("num_h", 5),
    ("mass", 1),
    ("formal_charge", 5),
    ("explicit_valency", 6),  # GroupGAT
    ("hybridization", 7),
    ("aromatic", 1),
    ("chiral", 4),
]
MERGED_BOND_NAMES = [
    ("bond_stereo", 6),  # GroupGAT: none, any, Z, E, Cis, Trans
    ("bond_type", 1),
    ("conjugated", 1),
    ("in_ring", 1),
    ("rotatable", 1),
]

# Default/RIGR: block-level only (RDKit opaque or RIGR spec)
DEFAULT_ATOM_NAMES = [("rdkit_atom_features", N_ATOM)]
DEFAULT_BOND_NAMES = [("stereo", 1), ("bond_type", 1), ("conjugated", 1), ("rotatable", 1)]
RIGR_ATOM_NAMES = [("atomic_num", 54), ("degree", 6), ("num_h", 5), ("mass", 1)]
RIGR_BOND_NAMES = [("in_ring", 1)]


def get_atom_feature_names(mode: FeaturizerMode) -> list[tuple[str, int]]:
    """Return [(feature_block_name, size), ...] for the given mode."""
    if mode == "default":
        return DEFAULT_ATOM_NAMES
    if mode == "rigr":
        return RIGR_ATOM_NAMES
    return MERGED_ATOM_NAMES


def get_bond_feature_names(mode: FeaturizerMode) -> list[tuple[str, int]]:
    """Return [(feature_block_name, size), ...] for the given mode."""
    if mode == "default":
        return DEFAULT_BOND_NAMES
    if mode == "rigr":
        return RIGR_BOND_NAMES
    return MERGED_BOND_NAMES


def save_feature_names_json(path: str | Path, mode: FeaturizerMode) -> None:
    """Save atom and bond feature names to JSON for interpretability and exports."""
    data = {
        "mode": mode,
        "atom_features": [{"name": n, "size": s} for n, s in get_atom_feature_names(mode)],
        "bond_features": [{"name": n, "size": s} for n, s in get_bond_feature_names(mode)],
        "groupgat_reference": {
            "atom": [{"name": n, "size": s} for n, s in GROUPGAT_ATOM_NAMES],
            "bond": [{"name": n, "size": s} for n, s in GROUPGAT_BOND_NAMES],
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_atom_dim(mode: FeaturizerMode) -> int:
    if mode == "default":
        return N_ATOM
    if mode == "rigr":
        return RIGR_ATOM_DIM
    return MERGED_ATOM_DIM


def get_bond_dim(mode: FeaturizerMode) -> int:
    if mode == "default":
        return N_BOND
    if mode == "rigr":
        return RIGR_BOND_DIM
    return MERGED_BOND_DIM


# --- Atom feature extractors (semantic blocks) ---

def _atom_default(atom) -> np.ndarray:
    return np.array(
        rdMolDescriptors.GetAtomFeatures(atom.GetOwningMol(), atom.GetIdx()),
        dtype=np.float32,
    )


def _atom_rigr(atom) -> np.ndarray:
    vec = np.zeros(RIGR_ATOM_DIM, dtype=np.float32)
    an = min(max(atom.GetAtomicNum(), 1), 54)
    vec[an - 1] = 1.0
    degree = min(atom.GetDegree(), 5)
    vec[54 + degree] = 1.0
    num_h = min(atom.GetTotalNumHs(), 4)
    vec[60 + num_h] = 1.0
    vec[65] = atom.GetMass() / 100.0
    return vec


def _atom_merged(atom) -> np.ndarray:
    """Compose unique semantics: RIGR base + GroupGAT extras (explicit_valency) + RDKit chiral."""
    vec = np.zeros(MERGED_ATOM_DIM, dtype=np.float32)
    off = 0
    an = min(max(atom.GetAtomicNum(), 1), 54)
    vec[off + (an - 1)] = 1.0
    off += 54
    degree = min(atom.GetDegree(), 5)
    vec[off + degree] = 1.0
    off += 6
    num_h = min(atom.GetTotalNumHs(), 4)
    vec[off + num_h] = 1.0
    off += 5
    vec[off] = atom.GetMass() / 100.0
    off += 1
    formal_charge = atom.GetFormalCharge()
    fc = max(min(formal_charge, 2), -2) + 2
    vec[off + fc] = 1.0
    off += 5
    # Explicit valency (GroupGAT): 0-5 one-hot
    expl_val = min(max(atom.GetValence(which=Chem.ValenceType.EXPLICIT), 0), 5)
    vec[off + expl_val] = 1.0
    off += 6
    hyb = atom.GetHybridization()
    hyb_map = {
        Chem.HybridizationType.SP: 0,
        Chem.HybridizationType.SP2: 1,
        Chem.HybridizationType.SP3: 2,
        Chem.HybridizationType.SP3D: 3,
        Chem.HybridizationType.SP3D2: 4,
    }
    vec[off + hyb_map.get(hyb, 6)] = 1.0
    off += 7
    vec[off] = float(atom.GetIsAromatic())
    off += 1
    chiral = atom.GetChiralTag()
    chiral_map = {
        Chem.ChiralType.CHI_UNSPECIFIED: 0,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
        Chem.ChiralType.CHI_OTHER: 3,
    }
    vec[off + chiral_map.get(chiral, 0)] = 1.0
    return vec


# --- Bond feature extractors ---

def _bond_default(mol) -> np.ndarray:
    bond_type_dict = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
    bond_stereo_dict = {"STEREONONE": 0, "STEREOANY": 1, "STEREOE": 2, "STEREOZ": 3}
    rotbonds = Lipinski._RotatableBonds(mol)
    feats = []
    for bond in mol.GetBonds():
        mol_obj = bond.GetOwningMol()
        ai, aj = sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        is_rot = (ai, aj) in rotbonds
        stereo = bond_stereo_dict[bond.GetStereo().name]
        btype = bond_type_dict[bond.GetBondType().name]
        conj = bond.GetIsConjugated()
        feats.append([stereo, btype, int(conj), int(is_rot)])
    return np.array(feats, dtype=np.float32) if feats else np.zeros((0, 4), dtype=np.float32)


def _bond_rigr(mol) -> np.ndarray:
    feats = [[float(bond.IsInRing())] for bond in mol.GetBonds()]
    return np.array(feats, dtype=np.float32) if feats else np.zeros((0, 1), dtype=np.float32)


def _bond_merged(mol) -> np.ndarray:
    """GroupGAT bond_stereo (6-d one-hot) + type + conj + in_ring + rotatable."""
    bond_type_dict = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
    # RDKit BondStereo: STEREONONE, STEREOANY, STEREOZ, STEREOE, STEREOCIS, STEREOTRANS -> 6-d one-hot
    BS = Chem.rdchem.BondStereo
    stereo_map = {
        BS.STEREONONE: 0,
        BS.STEREOANY: 1,
        BS.STEREOZ: 2,
        BS.STEREOE: 3,
        BS.STEREOCIS: 4,
        BS.STEREOTRANS: 5,
    }
    rotbonds = Lipinski._RotatableBonds(mol)
    feats = []
    for bond in mol.GetBonds():
        mol_obj = bond.GetOwningMol()
        ai, aj = sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        is_rot = (ai, aj) in rotbonds
        stereo_idx = stereo_map.get(bond.GetStereo(), 0)
        stereo_onehot = np.zeros(6, dtype=np.float32)
        stereo_onehot[stereo_idx] = 1.0
        btype = bond_type_dict[bond.GetBondType().name]
        conj = int(bond.GetIsConjugated())
        in_ring = float(bond.IsInRing())
        feat = np.concatenate([stereo_onehot, [btype, conj, in_ring, int(is_rot)]])
        feats.append(feat)
    return np.array(feats, dtype=np.float32) if feats else np.zeros((0, MERGED_BOND_DIM), dtype=np.float32)


def mol_to_graph(
    mol,
    mode: FeaturizerMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert mol to node_features, edge_index, edge_features.
    Edge features are duplicated for directed edges (a->b, b->a).
    """
    n_atom = get_atom_dim(mode)
    n_bond = get_bond_dim(mode)
    atom_fn = {"default": _atom_default, "rigr": _atom_rigr, "merged": _atom_merged}[mode]
    bond_fn = {"default": _bond_default, "rigr": _bond_rigr, "merged": _bond_merged}[mode]

    atom_feats = np.array([atom_fn(a) for a in mol.GetAtoms()], dtype=np.float32)
    bond_feats = bond_fn(mol)
    if bond_feats.size == 0:
        edge_index = np.zeros((2, 0), dtype=np.int32)
        edge_features = np.zeros((0, n_bond), dtype=np.float32)
        return atom_feats, edge_index, edge_features

    src_list, dst_list, feat_list = [], [], []
    for b, bond in enumerate(mol.GetBonds()):
        ai, aj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = bond_feats[b]
        src_list.extend([ai, aj])
        dst_list.extend([aj, ai])
        feat_list.append(f)
        feat_list.append(f)
    edge_index = np.stack([np.array(src_list), np.array(dst_list)], axis=0)
    edge_features = np.array(feat_list, dtype=np.float32)
    return atom_feats, edge_index, edge_features
