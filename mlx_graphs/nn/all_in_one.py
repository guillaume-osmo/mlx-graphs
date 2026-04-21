"""ChemeleonSMD-AllinOne fold-masked ensembles for blind cross-validation.

Paper-facing family name: SCORE-GNNs.

This module combines three ideas:

1. Independent experts with separate weights, similar in spirit to deep ensembles.
2. A mixture-style merge head over expert predictions, inspired by mixtures of experts
   and stacked generalization.
3. A fold-aware masking rule where expert ``k`` never receives supervised loss for
   samples assigned to holdout fold ``k``.

The key consequence is that, after training, selecting expert ``k`` for samples in
fold ``k`` yields an out-of-fold prediction by construction: that expert never saw
those targets in its own supervised loss.

References / inspiration:
- Jordan & Jacobs (1994), Hierarchical mixtures of experts.
- Wolpert (1992), Stacked generalization.
- Lakshminarayanan et al. (2017), Deep ensembles.

Notes
-----
``merged_prediction`` is a deployment-style ensemble prediction. It is useful on
new unseen data, but it should not be confused with the blind out-of-fold metric.
For blind CV accounting, use ``holdout_prediction`` and ``holdout_mse_loss``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear


def _as_prediction_matrix(predictions: mx.array) -> mx.array:
    """Normalize predictions to shape ``(num_graphs, num_experts)``."""
    if predictions.ndim == 1:
        return predictions[:, None]
    if predictions.ndim == 2:
        return predictions
    if predictions.ndim == 3 and predictions.shape[-1] == 1:
        return predictions[..., 0]
    raise ValueError(
        "Expected predictions with shape (N,), (N, K), or (N, K, 1). "
        f"Got {predictions.shape}."
    )


def _as_target_column(targets: mx.array) -> mx.array:
    """Normalize regression targets to shape ``(num_graphs, 1)``."""
    if targets.ndim == 1:
        return targets[:, None]
    if targets.ndim == 2 and targets.shape[-1] == 1:
        return targets
    raise ValueError(
        "Expected targets with shape (N,) or (N, 1). "
        f"Got {targets.shape}."
    )


def fold_holdout_mask(fold_ids: mx.array, num_experts: int) -> mx.array:
    """Return a boolean mask of active supervised pairs for ``(sample, expert)``.

    ``False`` marks the unique expert that must stay blind for that sample.
    """
    if num_experts < 2:
        raise ValueError("All-in-one CV requires at least two experts / folds.")
    fold_ids = fold_ids.astype(mx.int32).reshape(-1)
    expert_ids = mx.arange(num_experts, dtype=mx.int32)
    return fold_ids[:, None] != expert_ids[None, :]


def blind_holdout_predictions(expert_predictions: mx.array, fold_ids: mx.array) -> mx.array:
    """Select the blind out-of-fold prediction for each sample.

    For sample ``i`` assigned to fold ``k``, this returns expert ``k``'s prediction.
    """
    preds = _as_prediction_matrix(expert_predictions)
    mask = (~fold_holdout_mask(fold_ids, preds.shape[1])).astype(preds.dtype)
    return mx.sum(preds * mask, axis=1, keepdims=True)


def masked_expert_mse_loss(
    expert_predictions: mx.array,
    targets: mx.array,
    fold_ids: mx.array,
) -> mx.array:
    """Mean squared error over all non-held-out ``(sample, expert)`` pairs."""
    preds = _as_prediction_matrix(expert_predictions)
    y = _as_target_column(targets)
    weights = fold_holdout_mask(fold_ids, preds.shape[1]).astype(preds.dtype)
    sqerr = (preds - y) ** 2
    denom = mx.maximum(mx.sum(weights), mx.array(1.0, dtype=preds.dtype))
    return mx.sum(sqerr * weights) / denom


def holdout_mse_loss(
    expert_predictions: mx.array,
    targets: mx.array,
    fold_ids: mx.array,
) -> mx.array:
    """Mean squared error of the blind out-of-fold predictions."""
    holdout_pred = blind_holdout_predictions(expert_predictions, fold_ids)
    y = _as_target_column(targets)
    return mx.mean((holdout_pred - y) ** 2)


@dataclass
class AllInOneCVOutput:
    """Structured outputs for the fold-masked ensemble."""

    expert_predictions: mx.array
    gate_weights: mx.array
    merged_prediction: mx.array
    holdout_prediction: Optional[mx.array] = None


class AllInOneCVEnsembleRegressor(nn.Module):
    """Parallel experts with fold-masked supervision and a merge head.

    Parameters
    ----------
    experts:
        Sequence of expert regressors. Each expert must accept the same forward
        signature and return one prediction per graph.
    gate_hidden_dim:
        Hidden width of the merge/gating MLP.
    gate_feature_dim:
        Optional side-feature dimension concatenated to expert predictions before
        computing gate weights. This can be used for graph-level descriptors.
    gate_dropout:
        Dropout used inside the merge head.

    Training recipe
    ---------------
    1. Forward all experts on the same batch.
    2. Compute ``masked_expert_mse_loss`` with fold-aware masking.
    3. Track blind validation with ``holdout_prediction`` or ``holdout_mse_loss``.

    The default return value is:
    - ``holdout_prediction`` when ``fold_ids`` are provided.
    - ``merged_prediction`` otherwise.
    """

    def __init__(
        self,
        experts: Sequence[nn.Module],
        gate_hidden_dim: int = 32,
        gate_feature_dim: int = 0,
        gate_dropout: float = 0.0,
    ):
        super().__init__()
        if len(experts) < 2:
            raise ValueError("All-in-one CV requires at least two experts.")
        if gate_hidden_dim < 1:
            raise ValueError("gate_hidden_dim must be >= 1.")
        self.experts = list(experts)
        self.num_experts = len(self.experts)
        self.gate_feature_dim = gate_feature_dim
        gate_in_dim = self.num_experts + gate_feature_dim
        self.gate_fc1 = Linear(gate_in_dim, gate_hidden_dim)
        self.gate_fc2 = Linear(gate_hidden_dim, self.num_experts)
        self.gate_dropout = nn.Dropout(gate_dropout)

    @classmethod
    def from_factory(
        cls,
        num_experts: int,
        factory: Callable[[], nn.Module],
        gate_hidden_dim: int = 32,
        gate_feature_dim: int = 0,
        gate_dropout: float = 0.0,
    ) -> "AllInOneCVEnsembleRegressor":
        """Instantiate ``num_experts`` independent experts from a factory."""
        experts = [factory() for _ in range(num_experts)]
        return cls(
            experts,
            gate_hidden_dim=gate_hidden_dim,
            gate_feature_dim=gate_feature_dim,
            gate_dropout=gate_dropout,
        )

    def _stack_expert_predictions(
        self,
        *expert_args,
        training: bool = False,
        **expert_kwargs,
    ) -> mx.array:
        """Forward all experts and stack outputs as ``(num_graphs, num_experts)``."""
        kwargs = dict(expert_kwargs)
        kwargs["training"] = training
        outputs = []
        num_graphs = None
        for expert in self.experts:
            pred = _as_target_column(expert(*expert_args, **kwargs))
            if num_graphs is None:
                num_graphs = pred.shape[0]
            elif pred.shape[0] != num_graphs:
                raise ValueError("All experts must return the same number of graph predictions.")
            outputs.append(pred[:, 0])
        return mx.stack(outputs, axis=1)

    def _gate_weights(
        self,
        expert_predictions: mx.array,
        gate_features: Optional[mx.array],
        training: bool,
    ) -> mx.array:
        gate_in = expert_predictions
        if self.gate_feature_dim > 0:
            if gate_features is None:
                raise ValueError(
                    "gate_features must be provided when gate_feature_dim > 0."
                )
            if gate_features.ndim == 1:
                gate_features = gate_features[:, None]
            if gate_features.ndim != 2 or gate_features.shape[1] != self.gate_feature_dim:
                raise ValueError(
                    "gate_features must have shape "
                    f"(N, {self.gate_feature_dim}). Got {gate_features.shape}."
                )
            gate_in = mx.concatenate([gate_in, gate_features], axis=-1)
        elif gate_features is not None:
            raise ValueError(
                "Received gate_features but gate_feature_dim == 0. "
                "Set gate_feature_dim in the constructor to use side features."
            )
        hidden_in = self.gate_dropout(gate_in) if training else gate_in
        hidden = nn.leaky_relu(self.gate_fc1(hidden_in))
        hidden = self.gate_dropout(hidden) if training else hidden
        logits = self.gate_fc2(hidden)
        return mx.softmax(logits, axis=-1)

    def __call__(
        self,
        *expert_args,
        fold_ids: Optional[mx.array] = None,
        gate_features: Optional[mx.array] = None,
        training: bool = False,
        return_details: bool = False,
        **expert_kwargs,
    ) -> mx.array | AllInOneCVOutput:
        expert_predictions = self._stack_expert_predictions(
            *expert_args,
            training=training,
            **expert_kwargs,
        )
        gate_weights = self._gate_weights(
            expert_predictions, gate_features=gate_features, training=training
        )
        merged_prediction = mx.sum(
            gate_weights * expert_predictions, axis=1, keepdims=True
        )
        holdout_prediction = None
        if fold_ids is not None:
            holdout_prediction = blind_holdout_predictions(
                expert_predictions, fold_ids
            )
        if return_details:
            return AllInOneCVOutput(
                expert_predictions=expert_predictions,
                gate_weights=gate_weights,
                merged_prediction=merged_prediction,
                holdout_prediction=holdout_prediction,
            )
        if holdout_prediction is not None:
            return holdout_prediction
        return merged_prediction
