import mlx.core as mx
import pytest

from mlx_graphs.nn.conv.graph_ode_conv import (
    GraphODEAttFPRegressor,
    GraphODEDMPNNRegressor,
    GraphODEGATRegressor,
    GraphODERegressor,
)
from mlx_graphs.nn.conv.graph_score_conv import (
    GraphSCOREAttFPRegressor,
    GraphSCOREDMPNNRegressor,
    GraphSCOREGATRegressor,
    GraphSCORERegressor,
)
from mlx_graphs.nn.conv.groupgat_conv import GraphODEGroupGATRegressor
from mlx_graphs.nn.conv.graph_score_conv import GraphSCOREGroupGATRegressor
from mlx_graphs.nn.conv.mogat_conv import GraphODEMoGATRegressor
from mlx_graphs.nn.conv.graph_score_conv import GraphSCOREMoGATRegressor


def _make_graph_batch():
    node_dim = 8
    edge_dim = 4
    node_features = mx.random.uniform(0, 1, (12, node_dim))
    edge_features = mx.random.uniform(0, 1, (20, edge_dim))
    edge_index = mx.array(
        [
            [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 0, 2, 6, 6, 8, 11, 1, 4, 5, 9],
            [1, 2, 3, 4, 5, 0, 8, 9, 10, 11, 6, 6, 0, 2, 7, 10, 4, 1, 9, 5],
        ],
        dtype=mx.int32,
    )
    batch_indices = mx.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=mx.int32)
    group_features = mx.zeros((12, 3))
    return {
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "edge_index": edge_index,
        "node_features": node_features,
        "edge_features": edge_features,
        "batch_indices": batch_indices,
        "group_features": group_features,
    }


def _make_directed_batch():
    node_dim = 8
    edge_dim = 4
    node_features = mx.random.uniform(0, 1, (8, node_dim))
    directed_edge_index = mx.array(
        [
            [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6],
        ],
        dtype=mx.int32,
    )
    edge_reverse = mx.array([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10], dtype=mx.int32)
    edge_features = mx.random.uniform(0, 1, (directed_edge_index.shape[1], edge_dim))
    batch_indices = mx.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=mx.int32)
    return {
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "directed_edge_index": directed_edge_index,
        "edge_reverse": edge_reverse,
        "node_features": node_features,
        "edge_features": edge_features,
        "batch_indices": batch_indices,
    }


@pytest.mark.parametrize(
    ("score_cls", "ode_cls"),
    [
        (GraphSCORERegressor, GraphODERegressor),
        (GraphSCOREGATRegressor, GraphODEGATRegressor),
        (GraphSCOREAttFPRegressor, GraphODEAttFPRegressor),
        (GraphSCOREDMPNNRegressor, GraphODEDMPNNRegressor),
        (GraphSCOREGroupGATRegressor, GraphODEGroupGATRegressor),
        (GraphSCOREMoGATRegressor, GraphODEMoGATRegressor),
    ],
)
def test_graph_score_alias_identity(score_cls, ode_cls):
    assert score_cls is ode_cls


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("score", lambda d: GraphSCORERegressor(d["node_dim"], d["edge_dim"], hidden_dim=16, ode_steps=2)),
        (
            "score_gat",
            lambda d: GraphSCOREGATRegressor(
                d["node_dim"], d["edge_dim"], hidden_dim=16, heads=4, ode_steps=2
            ),
        ),
        (
            "score_attfp",
            lambda d: GraphSCOREAttFPRegressor(
                d["node_dim"], d["edge_dim"], hidden_dim=16, ode_steps=2
            ),
        ),
        (
            "score_groupgat",
            lambda d: GraphSCOREGroupGATRegressor(
                d["node_dim"], d["edge_dim"], n_groups=3, hidden_dim=16, ode_steps=2
            ),
        ),
        (
            "score_mogat",
            lambda d: GraphSCOREMoGATRegressor(
                d["node_dim"], d["edge_dim"], hidden_dim=16, ode_steps=2
            ),
        ),
    ],
)
def test_graph_score_regressors_smoke(name, factory):
    del name
    batch = _make_graph_batch()
    model = factory(batch)
    kwargs = {}
    if model.__class__ is GraphSCOREGroupGATRegressor:
        kwargs["group_features"] = batch["group_features"]
    out = model(
        batch["edge_index"],
        batch["node_features"],
        batch["edge_features"],
        batch["batch_indices"],
        training=False,
        **kwargs,
    )
    assert out.shape == (3, 1)
    assert mx.all(mx.isfinite(out)).item()


def test_graph_score_dmpnn_regressor_smoke():
    batch = _make_directed_batch()
    model = GraphSCOREDMPNNRegressor(
        batch["node_dim"], batch["edge_dim"], hidden_dim=16, ode_steps=2
    )
    out = model(
        batch["directed_edge_index"],
        batch["edge_reverse"],
        batch["node_features"],
        batch["edge_features"],
        batch["batch_indices"],
        training=False,
    )
    assert out.shape == (2, 1)
    assert mx.all(mx.isfinite(out)).item()
