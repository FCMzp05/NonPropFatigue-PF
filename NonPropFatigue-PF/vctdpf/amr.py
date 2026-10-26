from __future__ import annotations

import numpy as np

_BIG = np.int64(1) << 31


def _key(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.int64), b.astype(np.int64)
    return np.minimum(a, b)*_BIG+np.maximum(a, b)


def label_longest(nodes: np.ndarray, tris: np.ndarray) -> np.ndarray:
    x = nodes[tris]
    L = np.stack([np.hypot(*(x[:, 2]-x[:, 1]).T), np.hypot(*(x[:, 0]-x[:, 2]).T),
                  np.hypot(*(x[:, 1]-x[:, 0]).T)], axis=1)
    k = np.argmax(L, axis=1)
    idx = (k[:, None]+np.arange(3)) % 3
    return np.take_along_axis(tris, idx, axis=1)


def refine(nodes: np.ndarray, tris: np.ndarray, marked: np.ndarray):
    T = np.asarray(tris)
    E = np.stack([_key(T[:, 1], T[:, 2]), _key(T[:, 2], T[:, 0]), _key(T[:, 0], T[:, 1])], axis=1)
    edges = np.unique(E[np.asarray(marked, bool), 0])
    while True:
        grow = np.unique(E[np.isin(E, edges).any(1), 0])
        new = np.setdiff1d(grow, edges, assume_unique=True)
        if not len(new):
            break
        edges = np.union1d(edges, new)
    n0 = len(nodes)
    ea, eb = edges//_BIG, edges % _BIG
    nodes = np.vstack([nodes, 0.5*(nodes[ea]+nodes[eb])])
    parent = np.arange(len(T))
    while True:
        rk = _key(T[:, 1], T[:, 2])
        pos = np.searchsorted(edges, rk)
        pos = np.minimum(pos, len(edges)-1)
        sel = edges[pos] == rk if len(edges) else np.zeros(len(T), bool)
        if not sel.any():
            break
        m = n0+pos[sel]
        v0, v1, v2 = T[sel].T
        T = np.vstack([T[~sel], np.stack([m, v0, v1], 1), np.stack([m, v2, v0], 1)])
        parent = np.r_[parent[~sel], parent[sel], parent[sel]]
    return nodes, T, parent, ea, eb


def nodal(v: np.ndarray, ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    return np.concatenate([v, 0.5*(v[ea]+v[eb])])


def size(nodes: np.ndarray, tris: np.ndarray) -> np.ndarray:
    x = nodes[tris]
    u, w = x[:, 1]-x[:, 0], x[:, 2]-x[:, 0]
    return np.sqrt(np.abs(u[:, 0]*w[:, 1]-u[:, 1]*w[:, 0]))
