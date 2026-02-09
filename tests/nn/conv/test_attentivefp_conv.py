# Copyright © 2023-2024 Apple Inc.
# Unittests for AttentiveFP, AttentiveFPRegressor, AttentiveFPFlexibleRegressor.

import unittest

import mlx.core as mx

from mlx_graphs.nn.conv import (
    AttentiveFP,
    AttentiveFPFlexibleRegressor,
    AttentiveFPRegressor,
)


def _make_batch(n_atom: int, n_bond: int, num_nodes: int, num_edges: int, batch_size: int):
    """Create edge_index, node_features, edge_features, batch_indices."""
    mx.random.seed(0)
    # edge_index [2, num_edges]: src -> dst
    src = mx.random.randint(0, num_nodes, (num_edges,))
    dst = mx.random.randint(0, num_nodes, (num_edges,))
    edge_index = mx.stack([src, dst], axis=0)
    node_features = mx.random.normal((num_nodes, n_atom)).astype(mx.float32)
    edge_features = mx.random.normal((num_edges, n_bond)).astype(mx.float32)
    # batch_indices: assign each node to a graph [0, batch_size-1]
    nodes_per_graph = max(1, (num_nodes + batch_size - 1) // batch_size)
    batch_indices = mx.minimum(
        mx.arange(num_nodes, dtype=mx.int32) // nodes_per_graph,
        batch_size - 1,
    )
    return edge_index, node_features, edge_features, batch_indices


class TestAttentiveFP(unittest.TestCase):
    def test_attentivefp_forward_shape(self):
        n_atom, n_bond = 32, 8
        fp_dim, radius, T = 16, 2, 2
        num_nodes, num_edges, batch_size = 20, 50, 4
        model = AttentiveFP(
            n_atom, n_bond, fp_dim, radius, T, p_dropout=0.1, use_rms_norm=False
        )
        edge_index, node_features, edge_features, batch_indices = _make_batch(
            n_atom, n_bond, num_nodes, num_edges, batch_size
        )
        out = model(
            edge_index, node_features, edge_features, batch_indices, training=False
        )
        self.assertEqual(out.shape, (batch_size, 1), msg="AttentiveFP output shape")

    def test_attentivefp_regressor_forward_shape(self):
        n_atom, n_bond = 32, 8
        fp_dim, radius, T = 16, 2, 2
        num_nodes, num_edges, batch_size = 20, 50, 4
        model = AttentiveFPRegressor(
            n_atom, n_bond, fp_dim, radius, T, 0.1, use_rms_norm=True
        )
        edge_index, node_features, edge_features, batch_indices = _make_batch(
            n_atom, n_bond, num_nodes, num_edges, batch_size
        )
        out = model(
            edge_index, node_features, edge_features, batch_indices, training=False
        )
        self.assertEqual(
            out.shape, (batch_size, 1), msg="AttentiveFPRegressor output shape"
        )

    def test_attentivefp_regressor_training_mode(self):
        n_atom, n_bond = 16, 4
        fp_dim, radius, T = 8, 1, 1
        num_nodes, num_edges, batch_size = 10, 24, 2
        model = AttentiveFPRegressor(
            n_atom, n_bond, fp_dim, radius, T, 0.1, use_rms_norm=False
        )
        edge_index, node_features, edge_features, batch_indices = _make_batch(
            n_atom, n_bond, num_nodes, num_edges, batch_size
        )
        out_train = model(
            edge_index, node_features, edge_features, batch_indices, training=True
        )
        out_eval = model(
            edge_index, node_features, edge_features, batch_indices, training=False
        )
        self.assertEqual(out_train.shape, (batch_size, 1))
        self.assertEqual(out_eval.shape, (batch_size, 1))
        # With dropout, train and eval can differ; just check no nan
        self.assertTrue(mx.all(mx.isfinite(out_train)).item(), "train output finite")
        self.assertTrue(mx.all(mx.isfinite(out_eval)).item(), "eval output finite")

    def test_attentivefp_flexible_regressor_gru(self):
        n_atom, n_bond = 32, 8
        fp_dim, radius, T = 16, 2, 2
        num_nodes, num_edges, batch_size = 15, 40, 3
        model = AttentiveFPFlexibleRegressor(
            n_atom, n_bond, fp_dim, radius, T, 0.1, cell_type="gru"
        )
        edge_index, node_features, edge_features, batch_indices = _make_batch(
            n_atom, n_bond, num_nodes, num_edges, batch_size
        )
        out = model(
            edge_index, node_features, edge_features, batch_indices, training=False
        )
        self.assertEqual(
            out.shape, (batch_size, 1), msg="AttentiveFPFlexibleRegressor (gru) shape"
        )

    def test_attentivefp_flexible_regressor_microgru(self):
        n_atom, n_bond = 32, 8
        fp_dim, radius, T = 16, 2, 2
        num_nodes, num_edges, batch_size = 15, 40, 3
        model = AttentiveFPFlexibleRegressor(
            n_atom, n_bond, fp_dim, radius, T, 0.1, cell_type="microgru"
        )
        edge_index, node_features, edge_features, batch_indices = _make_batch(
            n_atom, n_bond, num_nodes, num_edges, batch_size
        )
        out = model(
            edge_index, node_features, edge_features, batch_indices, training=False
        )
        self.assertEqual(
            out.shape,
            (batch_size, 1),
            msg="AttentiveFPFlexibleRegressor (microgru) shape",
        )

    def test_attentivefp_flexible_regressor_invalid_cell_type(self):
        with self.assertRaises(ValueError):
            AttentiveFPFlexibleRegressor(
                32, 8, 16, 2, 2, 0.1, cell_type="invalid"
            )


if __name__ == "__main__":
    unittest.main()
