import numpy as np
from rdkit import Chem

from mlx_graphs.chem.group_contribution import (
    GroupContributionFeaturizer,
    augment_node_features_with_groups,
)


def test_group_contribution_featurizer_shapes():
    featurizer = GroupContributionFeaturizer()
    mol = Chem.MolFromSmiles("CCO")
    onehot = featurizer.mol_to_group_onehot(mol)

    assert onehot.shape[0] == mol.GetNumAtoms()
    assert onehot.shape[1] == featurizer.num_groups
    assert np.allclose(onehot.sum(axis=1), 1.0)


def test_augment_node_features_with_groups():
    featurizer = GroupContributionFeaturizer()
    mol = Chem.MolFromSmiles("CCO")
    node_features = np.ones((mol.GetNumAtoms(), 4), dtype=np.float32)
    augmented = augment_node_features_with_groups(node_features, mol, featurizer)

    assert augmented.shape == (mol.GetNumAtoms(), 4 + featurizer.num_groups)
