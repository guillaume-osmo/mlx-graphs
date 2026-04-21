import mlx.core as mx

from mlx_graphs.chem.mol_attfp_pooling import MolAttFPPooling


def test_mol_attfp_pooling_shape_and_finiteness():
    pool = MolAttFPPooling(hidden_dim=8, num_steps=2, dropout=0.1)
    x = mx.random.uniform(0, 1, (4, 6, 8))
    out = pool(x, training=True)

    assert out.shape == (4, 8)
    assert mx.all(mx.isfinite(out)).item()
