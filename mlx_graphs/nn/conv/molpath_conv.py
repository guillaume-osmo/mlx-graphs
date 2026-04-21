from __future__ import annotations

from collections import OrderedDict, deque
from typing import Optional
import math
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.mol_attention_readout import MolAttentionReadout
from mlx_graphs.utils import BMSSPConfig, bounded_sssp_nonneg, edge_list_to_adj, scatter


class MolPathRegressor(nn.Module):
    """MolPath-style regressor with shortest-path-aware message passing.

    Practical MLX implementation choices:
    - Bounded shortest-path expansion up to `max_hops` (frontier-limited BFS).
    - Cached path index tensors per batch structure to reduce repeated CPU work.
    - Distance-conditioned message transforms with learnable per-hop gates.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,  # kept for API symmetry with other regressors
        hidden_dim: int = 128,
        depth: int = 3,
        max_hops: int = 4,
        dropout: float = 0.1,
        mlp_units: tuple[int, ...] = (128, 64, 32),
        rdkit_dim: int = 0,
        mol_attention_steps: int = 0,
        mol_integrator: str = "euler",
        sssp_backend: str = "bfs",
        bmssp_block_size: int = 64,
        bmssp_outdegree_cap: int = 0,
        cache_size: int = 128,
    ):
        super().__init__()
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.max_hops = max_hops
        self.rdkit_dim = rdkit_dim
        self.cache_size = cache_size
        if sssp_backend not in ("bfs", "bmssp"):
            raise ValueError("sssp_backend must be one of ('bfs', 'bmssp')")
        self.sssp_backend = sssp_backend
        self.bmssp_cfg = BMSSPConfig(
            block_size=bmssp_block_size,
            use_block_frontier=True,
            outdegree_cap=bmssp_outdegree_cap,
        )

        self.node_proj = Linear(node_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        for l in range(depth):
            for d in range(1, max_hops + 1):
                setattr(self, f"path_lin_{l}_{d}", Linear(hidden_dim, hidden_dim))
                setattr(self, f"path_gate_{l}_{d}", mx.array([0.0]))

        if mol_attention_steps > 0:
            self.mol_readout = MolAttentionReadout(
                hidden_dim, mol_attention_steps, dropout, integrator=mol_integrator
            )
        else:
            self.mol_readout = None

        mlp_in = hidden_dim + (rdkit_dim if rdkit_dim > 0 else 0)
        units = [mlp_in] + list(mlp_units) + [1]
        self.mlp_layers = [Linear(units[i], units[i + 1], bias=True) for i in range(len(units) - 1)]

        # Batch-structure -> (src_idx, dst_idx, hop_dist) cache
        self._path_cache: OrderedDict[tuple, tuple[mx.array, mx.array, mx.array]] = OrderedDict()
        self._path_calls = 0
        self._path_cache_hits = 0
        self._path_cache_misses = 0
        self._path_build_time_s = 0.0

    def _cache_get(self, key: tuple) -> Optional[tuple[mx.array, mx.array, mx.array]]:
        item = self._path_cache.get(key)
        if item is None:
            return None
        self._path_cache.move_to_end(key)
        return item

    def _cache_put(self, key: tuple, value: tuple[mx.array, mx.array, mx.array]) -> None:
        self._path_cache[key] = value
        self._path_cache.move_to_end(key)
        while len(self._path_cache) > self.cache_size:
            self._path_cache.popitem(last=False)

    def _build_shortest_path_edges(
        self,
        edge_index: mx.array,
        batch_indices: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        self._path_calls += 1
        e_np = np.array(edge_index)
        b_np = np.array(batch_indices)
        key = (
            self.max_hops,
            self.sssp_backend,
            self.bmssp_cfg.block_size,
            self.bmssp_cfg.outdegree_cap,
            int(b_np.shape[0]),
            int(e_np.shape[1]),
            b_np.tobytes(),
            e_np.tobytes(),
        )
        cached = self._cache_get(key)
        if cached is not None:
            self._path_cache_hits += 1
            return cached
        self._path_cache_misses += 1
        t0 = time.perf_counter()

        n_nodes = b_np.shape[0]
        src_all, dst_all = e_np[0], e_np[1]
        num_graphs = int(b_np.max()) + 1 if b_np.size > 0 else 0

        src_list: list[int] = []
        dst_list: list[int] = []
        hop_list: list[int] = []

        for g in range(num_graphs):
            nodes = np.where(b_np == g)[0]
            if nodes.size == 0:
                continue
            node_set = set(nodes.tolist())
            adj = {int(u): [] for u in nodes.tolist()}
            # Build undirected adjacency inside each graph
            for u, v in zip(src_all.tolist(), dst_all.tolist()):
                if u in node_set and v in node_set and u != v:
                    adj[int(u)].append(int(v))
                    adj[int(v)].append(int(u))

            if self.sssp_backend == "bfs":
                for s in nodes.tolist():
                    visited = {int(s): 0}
                    q = deque([int(s)])
                    while q:
                        u = q.popleft()
                        d_u = visited[u]
                        if d_u >= self.max_hops:
                            continue
                        for v in adj[u]:
                            if v not in visited:
                                visited[v] = d_u + 1
                                q.append(v)
                    for v, d in visited.items():
                        if 1 <= d <= self.max_hops:
                            src_list.append(int(s))
                            dst_list.append(int(v))
                            hop_list.append(int(d))
            else:
                edges_g = [(u, v, 1.0) for u in nodes.tolist() for v in adj[int(u)]]
                adj_w = edge_list_to_adj(
                    n_nodes=n_nodes,
                    edges=edges_g,
                    undirected=False,
                )
                bound = float(self.max_hops + 1)
                for s in nodes.tolist():
                    dist = bounded_sssp_nonneg(adj_w, int(s), bound, self.bmssp_cfg)
                    for v in nodes.tolist():
                        if v == s:
                            continue
                        d = dist[int(v)]
                        if math.isfinite(d):
                            dd = int(round(d))
                            if 1 <= dd <= self.max_hops:
                                src_list.append(int(s))
                                dst_list.append(int(v))
                                hop_list.append(dd)

        if len(src_list) == 0:
            empty = mx.array(np.zeros((0,), dtype=np.int32))
            out = (empty, empty, empty)
            self._cache_put(key, out)
            self._path_build_time_s += time.perf_counter() - t0
            return out

        src_idx = mx.array(np.array(src_list, dtype=np.int32))
        dst_idx = mx.array(np.array(dst_list, dtype=np.int32))
        hop_idx = mx.array(np.array(hop_list, dtype=np.int32))
        out = (src_idx, dst_idx, hop_idx)
        self._cache_put(key, out)
        self._path_build_time_s += time.perf_counter() - t0
        return out

    def get_path_timing_stats(self) -> dict[str, float | int | str]:
        """Return shortest-path precompute/cache stats for this model instance."""
        avg_miss_s = (
            self._path_build_time_s / self._path_cache_misses
            if self._path_cache_misses > 0
            else 0.0
        )
        return {
            "backend": self.sssp_backend,
            "calls": self._path_calls,
            "cache_hits": self._path_cache_hits,
            "cache_misses": self._path_cache_misses,
            "build_time_s": self._path_build_time_s,
            "avg_miss_s": avg_miss_s,
        }

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,  # kept for API symmetry with trainer
        batch_indices: mx.array,
        graph_features: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        del edge_features  # intentionally unused in this shortest-path variant
        h = self.node_proj(node_features)
        src_idx, dst_idx, hop_idx = self._build_shortest_path_edges(edge_index, batch_indices)

        if src_idx.shape[0] > 0:
            num_nodes = h.shape[0]
            hop_np = np.array(hop_idx)
            for l in range(self.depth):
                agg = mx.zeros((num_nodes, self.hidden_dim))
                for d in range(1, self.max_hops + 1):
                    pos = np.where(hop_np == d)[0]
                    if pos.size == 0:
                        continue
                    pos_mx = mx.array(pos.astype(np.int32))
                    s_d = src_idx[pos_mx]
                    t_d = dst_idx[pos_mx]
                    lin = getattr(self, f"path_lin_{l}_{d}")
                    gate = mx.sigmoid(getattr(self, f"path_gate_{l}_{d}"))[0]
                    msg = gate * lin(h[s_d])
                    agg = agg + scatter(msg, t_d, out_size=num_nodes, aggr="add")
                h = nn.relu(h + agg)
                if training:
                    h = self.dropout(h)

        if self.mol_readout is not None:
            graph_repr = self.mol_readout(h, batch_indices, training=training)
        else:
            num_graphs = int(mx.max(batch_indices).item()) + 1
            graph_repr = scatter(h, batch_indices, out_size=num_graphs, aggr="add")

        if self.rdkit_dim > 0 and graph_features is not None:
            x = mx.concatenate([graph_repr, graph_features], axis=-1)
        else:
            x = graph_repr

        if training:
            x = self.dropout(x)
        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)
            if i < len(self.mlp_layers) - 1:
                x = nn.leaky_relu(x)
        return x
