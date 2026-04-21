import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.all_in_one import (
    AllInOneCVEnsembleRegressor,
    AllInOneCVOutput,
    blind_holdout_predictions,
    fold_holdout_mask,
    holdout_mse_loss,
    masked_expert_mse_loss,
)
from mlx_graphs.nn.conv.attentivefp_conv import AttentiveFPRegressor


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
    return {
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "edge_index": edge_index,
        "node_features": node_features,
        "edge_features": edge_features,
        "batch_indices": batch_indices,
        "graph_features": graph_features,
    }


class _ConstantExpert(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        batch_indices: mx.array,
        training: bool = False,
    ) -> mx.array:
        del edge_index, node_features, edge_features, training
        num_graphs = int(mx.max(batch_indices).item()) + 1
        return mx.full((num_graphs, 1), self.value)


def test_fold_holdout_mask_and_blind_selection():
    preds = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    fold_ids = mx.array([0, 2], dtype=mx.int32)
    mask = fold_holdout_mask(fold_ids, num_experts=3)
    expected = mx.array([[False, True, True], [True, True, False]])
    assert mx.all(mask == expected).item()

    blind = blind_holdout_predictions(preds, fold_ids)
    assert mx.allclose(blind, mx.array([[1.0], [6.0]])).item()


def test_masked_and_holdout_mse_match_manual_values():
    preds = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    targets = mx.array([[1.5], [5.0]])
    fold_ids = mx.array([0, 2], dtype=mx.int32)

    masked_loss = masked_expert_mse_loss(preds, targets, fold_ids)
    holdout_loss = holdout_mse_loss(preds, targets, fold_ids)

    assert abs(float(masked_loss.item()) - 0.875) < 1e-6
    assert abs(float(holdout_loss.item()) - 0.625) < 1e-6


def test_all_in_one_constant_experts_default_to_holdout_when_folds_are_given():
    batch = _make_graph_batch()
    model = AllInOneCVEnsembleRegressor(
        [_ConstantExpert(1.0), _ConstantExpert(2.0), _ConstantExpert(3.0)],
        gate_hidden_dim=8,
    )
    fold_ids = mx.array([0, 1, 2], dtype=mx.int32)

    out = model(
        batch["edge_index"],
        batch["node_features"],
        batch["edge_features"],
        batch["batch_indices"],
        fold_ids=fold_ids,
        training=False,
        return_details=True,
    )
    assert isinstance(out, AllInOneCVOutput)
    assert out.expert_predictions.shape == (3, 3)
    assert out.gate_weights.shape == (3, 3)
    assert out.merged_prediction.shape == (3, 1)
    assert out.holdout_prediction.shape == (3, 1)
    assert mx.allclose(out.holdout_prediction, mx.array([[1.0], [2.0], [3.0]])).item()
    assert mx.allclose(mx.sum(out.gate_weights, axis=1), mx.ones((3,))).item()

    default_out = model(
        batch["edge_index"],
        batch["node_features"],
        batch["edge_features"],
        batch["batch_indices"],
        fold_ids=fold_ids,
        training=False,
    )
    assert mx.allclose(default_out, out.holdout_prediction).item()


def test_all_in_one_factory_smoke_with_attentivefp_and_gate_features():
    batch = _make_graph_batch()
    model = AllInOneCVEnsembleRegressor.from_factory(
        num_experts=3,
        factory=lambda: AttentiveFPRegressor(
            batch["node_dim"],
            batch["edge_dim"],
            16,
            2,
            2,
            rdkit_dim=batch["graph_features"].shape[1],
        ),
        gate_hidden_dim=8,
        gate_feature_dim=batch["graph_features"].shape[1],
    )
    assert len({id(expert) for expert in model.experts}) == 3

    fold_ids = mx.array([0, 1, 2], dtype=mx.int32)
    out = model(
        batch["edge_index"],
        batch["node_features"],
        batch["edge_features"],
        batch["batch_indices"],
        graph_features=batch["graph_features"],
        gate_features=batch["graph_features"],
        fold_ids=fold_ids,
        training=False,
        return_details=True,
    )
    assert out.expert_predictions.shape == (3, 3)
    assert out.gate_weights.shape == (3, 3)
    assert out.merged_prediction.shape == (3, 1)
    assert out.holdout_prediction.shape == (3, 1)
    assert mx.all(mx.isfinite(out.expert_predictions)).item()
    assert mx.all(mx.isfinite(out.gate_weights)).item()
    assert mx.all(mx.isfinite(out.merged_prediction)).item()
    assert mx.all(mx.isfinite(out.holdout_prediction)).item()

    targets = mx.random.uniform(0, 1, (3, 1))
    loss = masked_expert_mse_loss(out.expert_predictions, targets, fold_ids)
    assert mx.all(mx.isfinite(loss)).item()
