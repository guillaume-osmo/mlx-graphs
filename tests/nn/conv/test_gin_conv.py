import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_graphs.nn.conv import GINERegressor
from mlx_graphs.nn.conv.gin_conv import GINConv


@pytest.mark.parametrize(
    "layer, edge_index, node_features, edge_weights, expected",
    [
        (
            GINConv(
                mlp=nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                ),
            ),
            mx.array([[0, 1, 2, 3], [0, 0, 1, 1]]),
            mx.random.uniform(0, 1, [10, 16]),
            None,
            (10, 32),
        ),
        (
            GINConv(
                mlp=nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                ),
            ),
            mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]]),
            mx.random.uniform(0, 1, [100, 16]),
            None,
            (100, 32),
        ),
        (
            GINConv(
                mlp=nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                ),
            ),
            mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]]),
            mx.random.uniform(0, 1, [100, 16]),
            mx.random.normal(
                [
                    5,
                ]
            ),
            (100, 32),
        ),
    ],
)
def test_gin_conv(layer, edge_index, node_features, edge_weights, expected):
    assert (
        expected == layer(edge_index, node_features, edge_weights=edge_weights).shape
    ), "GINConv failed"


@pytest.mark.parametrize(
    "layer, edge_index, node_features, edge_weights, edge_features, expected",
    [
        (
            GINConv(
                mlp=nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                ),
                node_features_dim=16,
                edge_features_dim=10,
            ),
            mx.array([[0, 1, 2, 3], [0, 0, 1, 1]]),
            mx.random.uniform(0, 1, [10, 16]),
            None,
            mx.random.normal((4, 10)),
            (10, 32),
        ),
        (
            GINConv(
                mlp=nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                ),
                node_features_dim=16,
                edge_features_dim=10,
            ),
            mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]]),
            mx.random.uniform(0, 1, [100, 16]),
            mx.random.normal(
                [
                    5,
                ]
            ),
            mx.random.normal((5, 10)),
            (100, 32),
        ),
    ],
)
def test_gine_conv(
    layer, edge_index, node_features, edge_weights, edge_features, expected
):
    assert (
        expected
        == layer(
            edge_index,
            node_features,
            edge_weights=edge_weights,
            edge_features=edge_features,
        ).shape
    ), "GINEConv failed"


def test_gine_regressor():
    regressor = GINERegressor(
        node_dim=8,
        edge_dim=4,
        hidden_dim=16,
        depth=2,
        dropout=0.1,
        rdkit_dim=6,
        mol_attention_steps=2,
        residual_dt=0.5,
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
    graph_features = mx.random.uniform(0, 1, [3, 6])

    y_hat = regressor(
        edge_index,
        node_features,
        edge_features,
        batch_indices,
        graph_features=graph_features,
        training=True,
    )

    assert y_hat.shape == (3, 1), "GINERegressor output shape failed"
    assert mx.all(mx.isfinite(y_hat)).item(), "GINERegressor output should be finite"
