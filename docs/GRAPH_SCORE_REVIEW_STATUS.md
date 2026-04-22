# Graph SCORE Review Status

Baseline for this review is `origin/main`.

This document uses the paper-facing `SCORE` naming.
The paper-facing import surface is now `mlx_graphs/nn/conv/graph_score_conv.py`,
which aliases the historical implementation in
`mlx_graphs/nn/conv/graph_ode_conv.py` to avoid breaking older imports.

## ODE -> SCORE naming

| Historical code naming | Paper / review naming |
| --- | --- |
| `graph_ode_conv.py` | `graph_score_conv.py` |
| `GraphODERegressor` family | `GraphSCORE` family |
| `ODE-GNNs` | `SCORE-GNNs` |

## Added Graph Conv Modules

| Paper name | Current code file | Review progress | Unit-tested | Valid code now | Known error | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `AttentiveFP` | `mlx_graphs/nn/conv/attentivefp_conv.py` | Dedicated layer/regressor review completed | Yes: dedicated conv tests + experimental smoke + MolAttFP pooling support tests | Yes | None currently | `Done` |
| `DMPNN` / `KA-DMPNN` | `mlx_graphs/nn/conv/dmpnn_conv.py` | Forward path and directed-edge helpers reviewed | Yes: experimental smoke coverage | Yes | No current failing test, but still needs dedicated per-layer assertions | `In progress` |
| `GraphSCORE` family | `mlx_graphs/nn/conv/graph_score_conv.py` | Alias surface reviewed; underlying `graph_ode_conv.py` implementation reviewed with supporting chem layers | Yes: experimental smoke + `graph_score` alias tests + `ode_block` + `ode_layers` + `rode_models` tests | Yes | No current failing test, but surface is large and still needs deeper per-variant review | `In progress` |
| `GraphTransformer` | `mlx_graphs/nn/conv/graph_transformer_conv.py` | Regressor path reviewed | Yes: experimental smoke coverage | Yes | No current failing test, but no dedicated block-level test file yet | `In progress` |
| `GroupGAT` / `GraphSCORE-GroupGAT` | `mlx_graphs/nn/conv/groupgat_conv.py` | Regressor path reviewed | Yes: experimental smoke coverage | Yes | No current failing test, but no dedicated unit test file yet | `In progress` |
| `KA-GCN` / `KA-GAT` / `GraphSCORE-KA-*` | `mlx_graphs/nn/conv/ka_gnn_conv.py` | Regressor path reviewed | Yes: experimental smoke coverage | Yes | No current failing test, but no dedicated per-layer test file yet | `In progress` |
| `MoGAT` / `GraphSCORE-MoGAT` | `mlx_graphs/nn/conv/mogat_conv.py` | Regressor path reviewed | Yes: experimental smoke coverage | Yes | No current failing test, but no dedicated per-layer test file yet | `In progress` |
| `MolPath` | `mlx_graphs/nn/conv/molpath_conv.py` | Regressor path and path-backend dependencies reviewed | Yes: experimental smoke coverage for `bfs` and `bmssp` + `tests/utils/test_bmssp.py` | Yes | None currently | `In progress` |
| `MPNN` | `mlx_graphs/nn/conv/mpnn_conv.py` | Regressor path reviewed | Yes: experimental smoke coverage | Yes | No current failing test, but no dedicated conv test file yet | `In progress` |

## Modified Upstream Conv Modules

| Module | Review progress | Unit-tested | Valid code now | Known error | Status |
| --- | --- | --- | --- | --- | --- |
| `gcn_conv.py` | Upstream-diff reviewed | Yes: `tests/nn/conv/test_gcn_conv.py` | Yes | None currently | `Done` |
| `gat_conv.py` | Upstream-diff reviewed | Yes: `tests/nn/conv/test_gat_conv.py` | Yes | None currently | `Done` |
| `gatv2_conv.py` | Upstream-diff reviewed | Yes: `tests/nn/conv/test_gatv2_conv.py` | Yes | None currently | `Done` |
| `gin_conv.py` | Upstream-diff reviewed | Yes: `tests/nn/conv/test_gin_conv.py` | Yes | None currently | `Done` |

## Targeted Test Evidence

These targeted review runs passed on the current `SCORE-GNNs` branch:

| Test command scope | Result |
| --- | --- |
| `tests/nn/conv/test_attentivefp_conv.py` + `tests/nn/conv/test_experimental_models.py` | `35 passed` |
| `tests/nn/conv/test_graph_score_conv.py` | `12 passed` |
| `tests/chem/test_mol_attfp_pooling.py` + `tests/chem/test_ode_block.py` + `tests/chem/test_ode_layers.py` + `tests/chem/test_tokenizer.py` + `tests/chem/test_rode_models.py` + `tests/chem/test_group_contribution.py` + `tests/chem/test_mol_featurizer.py` | `19 passed` |
| `tests/nn/conv/test_gcn_conv.py` + `tests/nn/conv/test_gat_conv.py` + `tests/nn/conv/test_gatv2_conv.py` + `tests/nn/conv/test_gin_conv.py` | `12 passed` |
| `tests/utils/test_bmssp.py` | `3 passed` |

## Meaning of Status

| Status | Meaning |
| --- | --- |
| `Done` | Targeted review finished and direct tests are in place for the module surface we changed. |
| `In progress` | Current code is passing the targeted smoke/support tests, but the module still needs more dedicated unit coverage or deeper variant-by-variant review before calling it fully closed. |
