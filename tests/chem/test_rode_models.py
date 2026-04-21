import mlx.core as mx
import pytest

from mlx_graphs.chem.rode_models import (
    CNF_MLP,
    CNF_MLP_NFP,
    CNF_RODE_MLP,
    RODE_SmilesX_MLP,
    RODE_TextCNN_MLP,
    SmilesX_MLP,
    TextCNN_MLP,
)
from mlx_graphs.chem.tokenizer import AtomwiseTokenizer, SmilesXTokenizer


def _token_batches():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    smilesx = SmilesXTokenizer(maxlen=16).add_molecules(smiles)
    atomwise = AtomwiseTokenizer(maxlen=16).add_molecules(smiles)
    return (
        mx.array(smilesx.encode_batch(smiles)),
        mx.array(atomwise.encode_batch(smiles)),
        smilesx.vocab_size,
        atomwise.vocab_size,
    )


@pytest.mark.parametrize(
    ("name", "factory", "use_atomwise"),
    [
        (
            "rode_smilesx",
            lambda sv, av: RODE_SmilesX_MLP(
                vocab_size=sv, maxlen=16, embedding_dim=8, lstm_units=8, n_ode_steps=2, pool_steps=1
            ),
            False,
        ),
        (
            "smilesx",
            lambda sv, av: SmilesX_MLP(
                vocab_size=sv, maxlen=16, embedding_dim=8, lstm_units=8, hidden_mlp=16
            ),
            False,
        ),
        (
            "rode_textcnn",
            lambda sv, av: RODE_TextCNN_MLP(
                vocab_size=av, maxlen=16, embedding_dim=8, conv_filters=8, kernel_sizes=(2, 3), n_ode_steps=2, pool_steps=1
            ),
            True,
        ),
        (
            "textcnn",
            lambda sv, av: TextCNN_MLP(
                vocab_size=av, maxlen=16, embedding_dim=8, conv_filters=8, kernel_sizes=(2, 3)
            ),
            True,
        ),
        (
            "cnf",
            lambda sv, av: CNF_MLP(
                vocab_size=av, maxlen=16, embedding_dim=8, conv_filters=8, kernel_sizes=(2, 3)
            ),
            True,
        ),
        (
            "cnf_rode",
            lambda sv, av: CNF_RODE_MLP(
                vocab_size=av, maxlen=16, embedding_dim=8, conv_filters=8, kernel_sizes=(2, 3)
            ),
            True,
        ),
        (
            "cnf_nfp",
            lambda sv, av: CNF_MLP_NFP(
                vocab_size=av, maxlen=16, embedding_dim=8, n_filters=3, fp_dim=8, mlp_dim=16
            ),
            True,
        ),
    ],
)
def test_rode_and_cnf_model_smoke(name, factory, use_atomwise):
    del name
    smilesx_batch, atomwise_batch, smilesx_vocab, atomwise_vocab = _token_batches()
    batch = atomwise_batch if use_atomwise else smilesx_batch
    model = factory(smilesx_vocab, atomwise_vocab)
    out = model(batch, training=True)

    assert out.shape == (3, 1)
    assert mx.all(mx.isfinite(out)).item()
