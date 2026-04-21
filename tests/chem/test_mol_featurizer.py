import json

from rdkit import Chem

from mlx_graphs.chem.mol_featurizer import (
    get_atom_dim,
    get_bond_dim,
    mol_to_graph,
    save_feature_names_json,
)


def test_mol_to_graph_shapes_across_modes():
    mol = Chem.MolFromSmiles("CCO")
    for mode in ("default", "rigr", "merged"):
        atom_features, edge_index, edge_features = mol_to_graph(mol, mode)
        assert atom_features.shape == (mol.GetNumAtoms(), get_atom_dim(mode))
        assert edge_index.shape == (2, 4)
        assert edge_features.shape == (4, get_bond_dim(mode))


def test_mol_to_graph_handles_edgeless_molecule():
    mol = Chem.MolFromSmiles("[He]")
    atom_features, edge_index, edge_features = mol_to_graph(mol, "merged")
    assert atom_features.shape[0] == 1
    assert edge_index.shape == (2, 0)
    assert edge_features.shape == (0, get_bond_dim("merged"))


def test_save_feature_names_json(tmp_path):
    path = tmp_path / "feature_names.json"
    save_feature_names_json(path, "merged")
    data = json.loads(path.read_text())

    assert data["mode"] == "merged"
    assert "atom_features" in data
    assert "bond_features" in data
