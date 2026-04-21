from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import bisect
import heapq
import math


@dataclass(frozen=True)
class BMSSPConfig:
    """Practical BMSSP-style bounded SSSP config.

    This implementation keeps exact distances for non-negative weights and
    uses a deterministic block-frontier extraction strategy inspired by BMSSP.
    """

    block_size: int = 64
    use_block_frontier: bool = True
    outdegree_cap: int = 0  # 0 means disabled


def _cap_outdegree(adj: list[list[tuple[int, float]]], cap: int) -> list[list[tuple[int, float]]]:
    if cap <= 0:
        return adj
    out: list[list[tuple[int, float]]] = []
    for nbrs in adj:
        if len(nbrs) <= cap:
            out.append(nbrs)
        else:
            # Keep lightest edges to preserve short paths as much as possible.
            out.append(sorted(nbrs, key=lambda x: x[1])[:cap])
    return out


def bounded_sssp_nonneg(
    adj: list[list[tuple[int, float]]],
    source: int,
    bound: float,
    cfg: BMSSPConfig,
) -> list[float]:
    """Exact bounded SSSP for non-negative weights.

    Distances >= bound are kept as +inf.
    """
    n = len(adj)
    if source < 0 or source >= n:
        raise ValueError("source out of range")
    if bound <= 0:
        return [math.inf] * n

    g = _cap_outdegree(adj, cfg.outdegree_cap)
    dist = [math.inf] * n
    dist[source] = 0.0

    if not cfg.use_block_frontier:
        pq: list[tuple[float, int]] = [(0.0, source)]
        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u != dist[u]:
                continue
            if d_u >= bound:
                continue
            for v, w in g[u]:
                if w < 0:
                    raise ValueError("negative edge weight is not supported")
                nd = d_u + w
                if nd < dist[v] and nd < bound:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    # Deterministic block frontier:
    # - maintain sorted block minima
    # - extract up to block_size smallest candidates per round
    # - relax in batch, reinsert improved vertices.
    blocks: list[list[tuple[float, int]]] = [[(0.0, source)]]
    mins: list[float] = [0.0]

    def push(item: tuple[float, int]) -> None:
        d, _ = item
        i = bisect.bisect_left(mins, d)
        if i == len(blocks):
            blocks.append([item])
            mins.append(d)
            return
        blk = blocks[i]
        blk.append(item)
        if d < mins[i]:
            mins[i] = d
        if len(blk) > 2 * max(8, cfg.block_size):
            blk.sort(key=lambda x: x[0])
            mid = len(blk) // 2
            left = blk[:mid]
            right = blk[mid:]
            blocks[i] = left
            mins[i] = left[0][0]
            blocks.insert(i + 1, right)
            mins.insert(i + 1, right[0][0])

    def pull_batch() -> list[tuple[float, int]]:
        if not blocks:
            return []
        out: list[tuple[float, int]] = []
        while blocks and len(out) < cfg.block_size:
            # choose block with global smallest min
            i = min(range(len(mins)), key=lambda x: mins[x])
            blk = blocks[i]
            blk.sort(key=lambda x: x[0])
            take = min(cfg.block_size - len(out), len(blk))
            out.extend(blk[:take])
            rem = blk[take:]
            if rem:
                blocks[i] = rem
                mins[i] = rem[0][0]
            else:
                blocks.pop(i)
                mins.pop(i)
        return out

    while blocks:
        batch = pull_batch()
        if not batch:
            break
        for d_u, u in batch:
            if d_u != dist[u]:
                continue
            if d_u >= bound:
                continue
            for v, w in g[u]:
                if w < 0:
                    raise ValueError("negative edge weight is not supported")
                nd = d_u + w
                if nd < dist[v] and nd < bound:
                    dist[v] = nd
                    push((nd, v))
    return dist


def edge_list_to_adj(
    n_nodes: int,
    edges: Iterable[tuple[int, int, float]],
    undirected: bool = True,
) -> list[list[tuple[int, float]]]:
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    for u, v, w in edges:
        if u < 0 or u >= n_nodes or v < 0 or v >= n_nodes:
            continue
        adj[u].append((v, float(w)))
        if undirected and u != v:
            adj[v].append((u, float(w)))
    return adj
