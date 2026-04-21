import math

import pytest

from mlx_graphs.utils.bmssp import BMSSPConfig, bounded_sssp_nonneg, edge_list_to_adj


def test_bounded_sssp_matches_dijkstra_mode():
    adj = edge_list_to_adj(
        4,
        [
            (0, 1, 1.0),
            (1, 2, 2.0),
            (0, 2, 5.0),
            (2, 3, 1.0),
        ],
        undirected=True,
    )
    cfg = BMSSPConfig(block_size=2, use_block_frontier=False)
    dist = bounded_sssp_nonneg(adj, source=0, bound=10.0, cfg=cfg)
    assert dist == [0.0, 1.0, 3.0, 4.0]


def test_bounded_sssp_block_frontier_matches_reference():
    adj = edge_list_to_adj(
        5,
        [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (0, 4, 10.0),
        ],
        undirected=True,
    )
    cfg = BMSSPConfig(block_size=2, use_block_frontier=True)
    dist = bounded_sssp_nonneg(adj, source=0, bound=5.0, cfg=cfg)
    assert dist[0] == 0.0
    assert dist[1] == 1.0
    assert dist[2] == 2.0
    assert dist[3] == 3.0
    assert dist[4] == 4.0


def test_bounded_sssp_respects_bound_and_negative_guard():
    adj = edge_list_to_adj(3, [(0, 1, 1.0), (1, 2, 2.0)], undirected=False)
    cfg = BMSSPConfig(block_size=2, use_block_frontier=True)
    dist = bounded_sssp_nonneg(adj, source=0, bound=2.0, cfg=cfg)
    assert dist[0] == 0.0
    assert dist[1] == 1.0
    assert math.isinf(dist[2])

    bad_adj = [[(1, -1.0)], [], []]
    with pytest.raises(ValueError):
        bounded_sssp_nonneg(bad_adj, source=0, bound=5.0, cfg=cfg)
