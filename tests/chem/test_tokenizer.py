from mlx_graphs.chem.tokenizer import (
    AtomwiseTokenizer,
    SmilesXTokenizer,
    get_smiles_tokenizer,
    get_smiles_tokenizer_atomwise,
    replace,
)


def test_replace_and_smilesx_tokenizer():
    assert replace("C@@H") == "C_H"
    tokens = get_smiles_tokenizer("CC(Cl)Br")
    assert "Cl" in tokens
    assert "Br" in tokens


def test_atomwise_tokenizer_and_encoding():
    atom_tokens = get_smiles_tokenizer_atomwise("CC(=O)O")
    assert atom_tokens
    assert "=" in atom_tokens

    smiles = ["CCO", "CC(=O)O"]
    smilesx = SmilesXTokenizer(maxlen=8).add_molecules(smiles)
    atomwise = AtomwiseTokenizer(maxlen=8).add_molecules(smiles)

    smilesx_batch = smilesx.encode_batch(smiles)
    atomwise_batch = atomwise.encode_batch(smiles)

    assert smilesx_batch.shape == (2, 8)
    assert atomwise_batch.shape == (2, 8)
    assert smilesx.vocab_size >= 2
    assert atomwise.vocab_size >= 2
