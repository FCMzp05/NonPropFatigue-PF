from __future__ import annotations

import numpy as np

_BIG = np.int64(1) << 31
_PAIRS = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


def _key(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.int64), b.astype(np.int64)
    return np.minimum(a, b)*_BIG+np.maximum(a, b)


def _edges(nodes: np.ndarray, T: np.ndarray):
    A, B = T[:, _PAIRS[:, 0]], T[:, _PAIRS[:, 1]]
    K = _key(A, B)
    L = np.linalg.norm(nodes[A]-nodes[B], axis=2)
    top = L >= L.max(1, keepdims=True)
    k = np.argmax(np.where(top, K, -1), axis=1)
    return K, k


def refine(nodes: np.ndarray, tets: np.ndarray, marked: np.ndarray):
    T = np.asarray(tets, dtype=np.int64)
    parent = np.arange(len(T))
    K, k = _edges(nodes, T)
    todo = np.unique(K[np.asarray(marked, bool), k[np.asarray(marked, bool)]])
    made_k = np.zeros(0, np.int64)
    made_i = np.zeros(0, np.int64)
    mids = []
    while len(todo):
        while True:
            Lk = K[np.arange(len(T)), k]
            has = np.isin(K, todo).any(1)
            new = np.setdiff1d(Lk[has], todo)
            if not len(new):
                break
            todo = np.union1d(todo, new)
        Lk = K[np.arange(len(T)), k]
        sel = np.isin(Lk, todo)
        split = np.unique(Lk[sel])
        fresh = split[~np.isin(split, made_k)]
        if len(fresh):
            n0 = len(nodes)
            ea, eb = fresh//_BIG, fresh % _BIG
            nodes = np.vstack([nodes, 0.5*(nodes[ea]+nodes[eb])])
            mids.append((ea, eb))
            made_k = np.r_[made_k, fresh]
            made_i = np.r_[made_i, n0+np.arange(len(fresh))]
            o = np.argsort(made_k)
            made_k, made_i = made_k[o], made_i[o]
        m = made_i[np.searchsorted(made_k, Lk[sel])]
        S = T[sel]
        loc = _PAIRS[k[sel]]
        r = np.arange(len(S))
        c1, c2 = S.copy(), S.copy()
        c1[r, loc[:, 1]] = m
        c2[r, loc[:, 0]] = m
        T = np.vstack([T[~sel], c1, c2])
        parent = np.r_[parent[~sel], parent[sel], parent[sel]]
        K, k = _edges(nodes, T)
        todo = todo[np.isin(todo, K)]
    return nodes, T, parent, mids


def nodal(v: np.ndarray, mids: list) -> np.ndarray:
    for ea, eb in mids:
        v = np.concatenate([v, 0.5*(v[ea]+v[eb])])
    return v


def locate(nodes: np.ndarray, tets: np.ndarray, pts: np.ndarray, k: int = 24, chunk: int = 50000):
    from scipy.spatial import cKDTree
    x = nodes[tets]
    M = np.linalg.inv(np.stack([x[:, 1]-x[:, 0], x[:, 2]-x[:, 0], x[:, 3]-x[:, 0]], axis=2))
    tree = cKDTree(x.mean(1))
    k = min(k, len(tets))
    el = np.empty(len(pts), np.int64)
    w = np.empty((len(pts), 4))
    for i0 in range(0, len(pts), chunk):
        p = pts[i0:i0+chunk]
        cand = tree.query(p, k=k)[1].reshape(len(p), k)
        l123 = np.einsum("nkij,nkj->nki", M[cand], p[:, None, :]-x[cand, 0])
        lam = np.concatenate([1.0-l123.sum(2, keepdims=True), l123], axis=2)
        best = np.argmax(lam.min(2), axis=1)
        r = np.arange(len(p))
        el[i0:i0+chunk] = cand[r, best]
        lb = np.clip(lam[r, best], 0.0, None)
        w[i0:i0+chunk] = lb/lb.sum(1, keepdims=True)
    return el, w


def size(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    x = nodes[tets]
    V = np.abs(np.einsum("ij,ij->i", x[:, 1]-x[:, 0], np.cross(x[:, 2]-x[:, 0], x[:, 3]-x[:, 0])))/6.0
    return (6.0*np.sqrt(2.0)*V)**(1.0/3.0)
