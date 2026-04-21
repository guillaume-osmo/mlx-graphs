# API Status and Literature Map

This file keeps the public API narrow while preserving the broader research work in-tree.

## Stable Public API

These names are re-exported from `mlx_graphs.nn.conv` and covered by the regular unit suite.

| Family | Public symbols | Validation | Primary literature |
| --- | --- | --- | --- |
| AttentiveFP | `AttentiveFP`, `AttentiveFPRegressor`, `AttentiveFPFlexibleRegressor` | shape, training, graph-feature, residual tests | AttentiveFP / graph-attention molecule readout |
| GAT | `GATConv`, `GATRegressor` | conv + regressor tests | Graph Attention Networks |
| GATv2 | `GATv2Conv`, `GATv2Regressor` | conv + regressor tests | How Attentive are Graph Attention Networks? |
| GCN | `GCNConv`, `GCNRegressor` | conv + regressor tests | Semi-Supervised Classification with Graph Convolutional Networks |
| GIN / GINE | `GINConv`, `GINERegressor` | conv + regressor tests | How Powerful are Graph Neural Networks? / GINE-style edge-conditioned extension |
| Generic convs | `GeneralizedRelationalConv`, `SAGEConv`, `SimpleConv` | existing legacy tests | internal mlx-graphs core layers |

## Experimental Direct-Module API

These modules are kept in-tree, smoke-tested, and importable directly, but are not re-exported from `mlx_graphs.nn.conv` yet.

| Module | Main classes | Current status | Primary literature / inspiration |
| --- | --- | --- | --- |
| `mlx_graphs.nn.conv.mpnn_conv` | `MPNNConv`, `MPNN`, `MPNNRegressor` | smoke-tested | Neural Message Passing for Quantum Chemistry |
| `mlx_graphs.nn.conv.dmpnn_conv` | `DMPNNConv`, `DMPNN`, `DMPNNRegressor`, `KADMPNN`, `KADMPNNRegressor` | smoke-tested with directed-edge helpers | ChemProp / Directed MPNN |
| `mlx_graphs.nn.conv.graph_transformer_conv` | `GraphTransformerBlock`, `GraphTransformer`, `GraphTransformerRegressor` | smoke-tested | graph transformer family |
| `mlx_graphs.nn.conv.ka_gnn_conv` | `KAGCNRegressor`, `KAGATRegressor` | smoke-tested | KAN-style nonlinearities composed with GCN/GAT |
| `mlx_graphs.nn.conv.groupgat_conv` | `GroupGATRegressor`, `GraphODEGroupGATRegressor` | smoke-tested | GC-GNN / GroupGAT |
| `mlx_graphs.nn.conv.mogat_conv` | `MoGATRegressor`, `GraphODEMoGATRegressor` | smoke-tested | Multi-order Graph Attention Network |
| `mlx_graphs.nn.conv.molpath_conv` | `MolPathRegressor` | smoke-tested, depends on `bmssp` utils | shortest-path-aware message passing |
| `mlx_graphs.nn.conv.graph_ode_conv` | `GraphODE*Regressor` family | representative variants smoke-tested | Neural ODE + backbone-specific message operators |
| `mlx_graphs.nn.all_in_one` | `AllInOneCVEnsembleRegressor`, masking helpers | direct unit tests | mixtures of experts + stacked generalization + deep ensembles |
| `mlx_graphs.nn.mol_attention_readout` | `MolAttentionReadout` | direct unit test + reused broadly | AttentiveFP-style graph readout |
| `mlx_graphs.utils.bmssp` | `BMSSPConfig`, `bounded_sssp_nonneg`, `edge_list_to_adj` | direct unit tests | bounded shortest-path frontier scheduling |

## Chemistry and Sequence Utilities

These are available from `mlx_graphs.chem` and now have dedicated smoke/unit coverage.

| Area | Exports | Validation | Notes |
| --- | --- | --- | --- |
| Molecular featurization | `mol_to_graph`, `get_atom_dim`, `get_bond_dim`, feature-name helpers | RDKit-backed tests | includes `default`, `rigr`, and `merged` modes |
| Group contribution features | `GroupContributionFeaturizer`, `augment_node_features_with_groups` | RDKit-backed tests | uses `MG_plus_reference.csv` in-tree |
| Tokenization | `SmilesXTokenizer`, `AtomwiseTokenizer`, tokenizer helpers | direct tests | `Cl` and `Br` are preserved as single tokens |
| Sequence pooling | `MolAttFPPooling` | direct tests | AttentiveFP-style readout for sequence encoders |
| ODE building blocks | `ODEBlock`, `ODESkipLayer`, `ODE_MLP`, `ODE_GRU`, `ODE_CNN` | direct tests | fixed MLX GRU and Conv1d layout assumptions |
| RODE / text models | `RODE_SmilesX_MLP`, `SmilesX_MLP`, `RODE_TextCNN_MLP`, `TextCNN_MLP`, `CNF_MLP`, `CNF_RODE_MLP`, `CNF_MLP_NFP` | direct smoke tests | research-oriented sequence baselines and ODE heads |

## Validation Snapshot

- Full local test suite: `160 passed`, `13 skipped`
- Added broad model-family smoke coverage under `tests/nn/conv/test_experimental_models.py`
- Added chemistry and shortest-path coverage under `tests/chem/` and `tests/utils/test_bmssp.py`
- Added fold-masked ensemble coverage under `tests/nn/test_all_in_one.py`

## Promotion Rules

Promote a direct-module API into `mlx_graphs.nn.conv` only when all of the following are true:

1. The constructor and forward signature are stable enough for external use.
2. The module has dedicated tests beyond a single import smoke.
3. The literature / inspiration is documented here.
4. The module does not rely on hidden side modules or half-exposed experimental flags.
