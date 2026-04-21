# ChemeleonSMD-AllinOne

Paper-facing family name: `SCORE-GNNs`.

This experimental module implements the `ChemeleonSMD-AllinOne` idea: train `K` independent
experts in parallel, assign each sample to one holdout fold, and never expose that
sample's target to its matching expert.

## Why

Classic `K`-fold CV trains `K` separate models. `ChemeleonSMD-AllinOne` keeps the same
blindness rule inside a single training job:

- expert `0` is blind to fold `0`
- expert `1` is blind to fold `1`
- ...
- expert `K-1` is blind to fold `K-1`

After training, the blind out-of-fold prediction for sample `i` is simply the output
of expert `fold_id[i]`.

## What Is Blind and What Is Not

- `holdout_prediction`: blind out-of-fold prediction by construction.
- `masked_expert_mse_loss`: the training loss that enforces the blindness rule.
- `merged_prediction`: a deployment-style mixture / stacking head over all experts.

The important caution is that `merged_prediction` is not the blind CV metric. Use it
for test-time ensembling on new data. Use `holdout_prediction` for CV accounting.

## API

Module path: `mlx_graphs.nn.all_in_one`

Stable code name:
- `AllInOneCVEnsembleRegressor`
- `AllInOneCVOutput`
- `fold_holdout_mask`
- `blind_holdout_predictions`
- `masked_expert_mse_loss`
- `holdout_mse_loss`

Paper-facing names:
- `ChemeleonSMD-AllinOne`
- `SCORE-GNNs`

## Example

```python
import mlx.core as mx
from mlx_graphs.nn.all_in_one import (
    AllInOneCVEnsembleRegressor,
    masked_expert_mse_loss,
    holdout_mse_loss,
)
from mlx_graphs.nn.conv.attentivefp_conv import AttentiveFPRegressor

num_folds = 5
model = AllInOneCVEnsembleRegressor.from_factory(
    num_experts=num_folds,
    factory=lambda: AttentiveFPRegressor(n_atom, n_bond, 128, 2, 2),
    gate_hidden_dim=32,
)

out = model(
    edge_index,
    node_features,
    edge_features,
    batch_indices,
    fold_ids=fold_ids,
    training=True,
    return_details=True,
)

train_loss = masked_expert_mse_loss(out.expert_predictions, targets, fold_ids)
oof_mse = holdout_mse_loss(out.expert_predictions, targets, fold_ids)
```

## Literature / Inspiration

This module is a practical combination rather than a literal port of one paper:

- Jordan & Jacobs (1994): hierarchical mixtures of experts
- Wolpert (1992): stacked generalization
- Lakshminarayanan et al. (2017): deep ensembles

The fold-masked supervision rule itself is an implementation trick for parallelized
out-of-fold training.
