import mlx.core as mx
import pytest

from mlx_graphs.nn.conv.attentivefp_conv import AttentiveFPRegressor
from mlx_graphs.nn.conv.dmpnn_conv import (
    DMPNNRegressor,
    KADMPNNRegressor,
    edge_reverse_from_directed_pairs,
)
from mlx_graphs.nn.conv.graph_ode_conv import (
    GraphODEAttFPRegressor,
    GraphODEDMPNNRegressor,
    GraphODEDiffGATRegressor,
    GraphODEDiffGINERegressor,
    GraphODEDiffGNNRegressor,
    GraphODEGATRegressor,
    GraphODEGATv2Regressor,
    GraphODEGCNRegressor,
    GraphODEGINERegressor,
    GraphODEGTRegressor,
    GraphODEKADMPNNRegressor,
    GraphODEKAGATRegressor,
    GraphODEKAGCNRegressor,
    GraphODERegressor,
)
from mlx_graphs.nn.conv.graph_transformer_conv import GraphTransformerRegressor
from mlx_graphs.nn.conv.groupgat_conv import GraphODEGroupGATRegressor, GroupGATRegressor
from mlx_graphs.nn.conv.ka_gnn_conv import KAGATRegressor, KAGCNRegressor
from mlx_graphs.nn.conv.mogat_conv import GraphODEMoGATRegressor, MoGATRegressor
from mlx_graphs.nn.conv.molpath_conv import MolPathRegressor
from mlx_graphs.nn.conv.mpnn_conv import MPNNRegressor
from mlx_graphs.nn.mol_attention_readout import MolAttentionReadout


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
    graph_features = mx.random.uniform(0, 1, (3, 5))
    group_features = mx.zeros((12, 3))
    return {
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "edge_index": edge_index,
        "node_features": node_features,
        "edge_features": edge_features,
        "batch_indices": batch_indices,
        "graph_features": graph_features,
        "group_features": group_features,
    }


def _make_directed_batch():
    node_dim = 8
    edge_dim = 4
    node_features = mx.random.uniform(0, 1, (8, node_dim))
    # Consecutive reverse-edge pairs so edge_reverse_from_directed_pairs works.
    directed_edge_index = mx.array(
        [
            [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6],
        ],
        dtype=mx.int32,
    )
    edge_reverse = edge_reverse_from_directed_pairs(directed_edge_index.shape[1])
    edge_features = mx.random.uniform(0, 1, (directed_edge_index.shape[1], edge_dim))
    batch_indices = mx.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=mx.int32)
    graph_features = mx.random.uniform(0, 1, (2, 5))
    return {
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "directed_edge_index": directed_edge_index,
        "edge_reverse": edge_reverse,
        "node_features": node_features,
        "edge_features": edge_features,
        "batch_indices": batch_indices,
        "graph_features": graph_features,
    }


def test_mol_attention_readout_smoke():
    readout = MolAttentionReadout(hidden_dim=16, num_steps=2, dropout=0.1)
    node_features = mx.random.uniform(0, 1, (9, 16))
    batch_indices = mx.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=mx.int32)
    out = readout(node_features, batch_indices, training=True)
    assert out.shape == (3, 16)
    assert mx.all(mx.isfinite(out)).item()


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("attentivefp", lambda d: AttentiveFPRegressor(d["node_dim"], d["edge_dim"], 16, 2, 2)),
        ("mpnn", lambda d: MPNNRegressor(d["node_dim"], d["edge_dim"], hidden_dim=16, depth=2)),
        (
            "graph_transformer",
            lambda d: GraphTransformerRegressor(
                d["node_dim"], d["edge_dim"], hidden_dim=16, num_heads=4, depth=2
            ),
        ),
        ("kagcn", lambda d: KAGCNRegressor(d["node_dim"], d["edge_dim"], hidden_dim=16, depth=2)),
        (
            "kagat",
            lambda d: KAGATRegressor(
                d["node_dim"], d["edge_dim"], hidden_dim=24, heads=3, depth=2
            ),
        ),
        ("mogat", lambda d: MoGATRegressor(d["node_dim"], d["edge_dim"], fingerprint_dim=16, radius=2)),
        (
            "molpath_bfs",
            lambda d: MolPathRegressor(
                d["node_dim"], d["edge_dim"], hidden_dim=16, depth=2, max_hops=3, sssp_backend="bfs"
            ),
        ),
        (
            "molpath_bmssp",
            lambda d: MolPathRegressor(
                d["node_dim"], d["edge_dim"], hidden_dim=16, depth=2, max_hops=3, sssp_backend="bmssp"
            ),
        ),
    ],
)
def test_additional_regressors_smoke(name, factory):
    del name
    batch = _make_graph_batch()
    model = factory(batch)
    out = model(
        batch["edge_index"],
        batch["node_features"],
        batch["edge_features"],
        batch["batch_indices"],
        training=False,
    )
    assert out.shape == (3, 1)
    assert mx.all(mx.isfinite(out)).item()


def test_groupgat_regressor_smoke():
    batch = _make_graph_batch()
    model = GroupGATRegressor(
        batch["node_dim"], batch["edge_dim"], n_groups=3, fingerprint_dim=16, radius=2, T=2
    )
    out = model(
        batch["edge_index"],
        batch["node_features"],
        batch["edge_features"],
        batch["batch_indices"],
        group_features=batch["group_features"],
        training=False,
    )
    assert out.shape == (3, 1)
    assert mx.all(mx.isfinite(out)).item()


def test_graph_ode_groupgat_regressor_smoke():
    batch = _make_graph_batch()
    model = GraphODEGroupGATRegressor(
        batch["node_dim"], batch["edge_dim"], n_groups=3, hidden_dim=16, ode_steps=2
    )
    out = model(
        batch["edge_index"],
        batch["node_features"],
        batch["edge_features"],
        batch["batch_indices"],
        group_features=batch["group_features"],
        training=False,
    )
    assert out.shape == (3, 1)
    assert mx.all(mx.isfinite(out)).item()


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("dmpnn", lambda d: DMPNNRegressor(d["node_dim"], d["edge_dim"], hidden_dim=16, depth=2)),
        (
            "kadmpnn",
            lambda d: KADMPNNRegressor(d["node_dim"], d["edge_dim"], hidden_dim=16, depth=2),
        ),
    ],
)
def test_dmpnn_regressors_smoke(name, factory):
    del name
    batch = _make_directed_batch()
    model = factory(batch)
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


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("graph_ode", lambda d: GraphODERegressor(d["node_dim"], d["edge_dim"], hidden_dim=16, ode_steps=2)),
        (
            "graph_ode_diff_gnn",
            lambda d: GraphODEDiffGNNRegressor(d["node_dim"], d["edge_dim"], hidden_dim=16, ode_steps=2),
        ),
        (
            "graph_ode_diff_gat",
            lambda d: GraphODEDiffGATRegressor(
                d["node_dim"], d["edge_dim"], hidden_dim=16, heads=4, ode_steps=2
            ),
        ),
        (
            "graph_ode_diff_gine",
            lambda d: GraphODEDiffGINERegressor(d["node_dim"], d["edge_dim"], hidden_dim=16, ode_steps=2),
        ),
        (
            "graph_ode_gcn",
            lambda d: GraphODEGCNRegressor(d["node_dim"], d["edge_dim"], hidden_dim=16, ode_steps=2),
        ),
        (
            "graph_ode_gat",
            lambda d: GraphODEGATRegressor(
                d["node_dim"], d["edge_dim"], hidden_dim=16, heads=4, ode_steps=2
            ),
        ),
        (
            "graph_ode_gatv2",
            lambda d: GraphODEGATv2Regressor(
                d["node_dim"], d["edge_dim"], hidden_dim=16, heads=4, ode_steps=2
            ),
        ),
        (
            "graph_ode_gine",
            lambda d: GraphODEGINERegressor(d["node_dim"], d["edge_dim"], hidden_dim=16, ode_steps=2),
        ),
        (
            "graph_ode_gt",
            lambda d: GraphODEGTRegressor(
                d["node_dim"], d["edge_dim"], hidden_dim=16, num_heads=4, ode_steps=2
            ),
        ),
        (
            "graph_ode_kagcn",
            lambda d: GraphODEKAGCNRegressor(d["node_dim"], d["edge_dim"], hidden_dim=16, ode_steps=2),
        ),
        (
            "graph_ode_kagat",
            lambda d: GraphODEKAGATRegressor(
                d["node_dim"], d["edge_dim"], hidden_dim=16, heads=4, ode_steps=2
            ),
        ),
        (
            "graph_ode_attfp",
            lambda d: GraphODEAttFPRegressor(d["node_dim"], d["edge_dim"], hidden_dim=16, ode_steps=2),
        ),
    ],
)
def test_graph_ode_regressors_smoke(name, factory):
    del name
    batch = _make_graph_batch()
    model = factory(batch)
    out = model(
        batch["edge_index"],
        batch["node_features"],
        batch["edge_features"],
        batch["batch_indices"],
        training=False,
    )
    assert out.shape == (3, 1)
    assert mx.all(mx.isfinite(out)).item()


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        (
            "graph_ode_dmpnn",
            lambda d: GraphODEDMPNNRegressor(d["node_dim"], d["edge_dim"], hidden_dim=16, ode_steps=2),
        ),
        (
            "graph_ode_kadmpnn",
            lambda d: GraphODEKADMPNNRegressor(d["node_dim"], d["edge_dim"], hidden_dim=16, ode_steps=2),
        ),
    ],
)
def test_graph_ode_dmpnn_regressors_smoke(name, factory):
    del name
    batch = _make_directed_batch()
    model = factory(batch)
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


def test_graph_ode_mogat_regressor_smoke():
    batch = _make_graph_batch()
    model = GraphODEMoGATRegressor(
        batch["node_dim"], batch["edge_dim"], hidden_dim=16, ode_steps=2
    )
    out = model(
        batch["edge_index"],
        batch["node_features"],
        batch["edge_features"],
        batch["batch_indices"],
        training=False,
    )
    assert out.shape == (3, 1)
    assert mx.all(mx.isfinite(out)).item()
