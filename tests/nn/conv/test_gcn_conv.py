import mlx.core as mx

from mlx_graphs.nn.conv import GCNConv, GCNRegressor

mx.random.seed(42)


def test_gcn_conv():
    conv = GCNConv(8, 20)

    node_features = mx.random.uniform(0, 1, [6, 8])
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    y_hat1 = conv(edge_index, node_features)

    node_features = mx.random.uniform(-1, 1, [6, 8])
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    y_hat2 = conv(edge_index, node_features)

    conv = GCNConv(16, 32)
    node_features = mx.random.uniform(0, 1, [100, 16])
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    y_hat3 = conv(edge_index, node_features)

    conv = GCNConv(16, 32, bias=False)
    node_features = mx.random.uniform(0, 1, [100, 16])
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    y_hat4 = conv(
        edge_index,
        node_features,
        edge_weights=mx.random.normal(
            [
                5,
            ]
        ),
    )

    conv = GCNConv(8, 16, add_self_loops=True)
    node_features = mx.random.uniform(0, 1, [6, 8])
    edge_index = mx.array([[0, 1, 2], [1, 2, 3]])
    y_hat5 = conv(edge_index, node_features)

    assert y_hat1.shape == (6, 20), "Simple GCNConv failed"
    assert y_hat2.shape == (6, 20), "GCNConv with negative values failed"
    assert y_hat3.shape == (100, 32), "GCNConv with different shapes failed"
    assert y_hat4.shape == (100, 32), "GCNConv without bias failed"
    assert y_hat5.shape == (6, 16), "GCNConv with self-loops failed"


def test_gcn_regressor():
    regressor = GCNRegressor(
        node_dim=8,
        edge_dim=4,
        hidden_dim=16,
        depth=3,
        dropout=0.1,
        rdkit_dim=7,
        mol_attention_steps=2,
        residual_dt="add",
    )
    node_features = mx.random.uniform(0, 1, [12, 8])
    edge_features = mx.random.uniform(0, 1, [20, 4])
    edge_index = mx.array(
        [
            [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 0, 2, 6, 6, 8, 11, 1, 4, 5, 9],
            [1, 2, 3, 4, 5, 0, 8, 9, 10, 11, 6, 6, 0, 2, 7, 10, 4, 1, 9, 5],
        ]
    )
    batch_indices = mx.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    graph_features = mx.random.uniform(0, 1, [3, 7])

    y_hat = regressor(
        edge_index,
        node_features,
        edge_features,
        batch_indices,
        graph_features=graph_features,
        training=True,
    )

    assert y_hat.shape == (3, 1), "GCNRegressor output shape failed"
    assert mx.all(mx.isfinite(y_hat)).item(), "GCNRegressor output should be finite"
