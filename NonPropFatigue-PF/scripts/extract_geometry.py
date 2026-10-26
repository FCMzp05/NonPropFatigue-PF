from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inp_reader import read_inp

QUAD8_EDGES = ((0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7))


def loops_from_mesh(m: dict) -> list[np.ndarray]:
    pos = {k: j for j, k in enumerate(m["node_ids"])}
    C = np.vectorize(pos.get)(m["conn"])
    X = m["X"]
    key = np.round(X*1e6).astype(np.int64)
    cid = {tuple(k): i for i, k in enumerate(key)}
    canon = np.array([cid[tuple(k)] for k in key])
    E = np.concatenate([C[:, list(e)] for e in QUAD8_EDGES])
    Ec = canon[E[:, :2]]
    s = np.sort(Ec, axis=1)
    u, inv, cnt = np.unique(s, axis=0, return_inverse=True, return_counts=True)
    bnd = E[cnt[inv] == 1]
    nxt = {}
    for a, b, mid in bnd:
        nxt[canon[a]] = (canon[b], mid)
    loops, seen = [], set()
    for start in list(nxt):
        if start in seen:
            continue
        pts, cur = [], start
        while cur not in seen:
            seen.add(cur)
            b, mid = nxt[cur]
            pts += [X[cur], X[mid]]
            cur = b
        loops.append(np.array(pts))
    return loops


def main():
    m = read_inp(Path(sys.argv[1]))
    loops = loops_from_mesh(m)
    area = [0.5*abs(np.dot(L[:, 0], np.roll(L[:, 1], -1))-np.dot(L[:, 1], np.roll(L[:, 0], -1))) for L in loops]
    o = int(np.argmax(area))
    holes = [L for i, L in enumerate(loops) if i != o and area[i] > 1.0]
    print(f"outer loop {len(loops[o])} pts area {area[o]:.0f}; {len(holes)} holes; "
          f"dropped {len(loops)-1-len(holes)} seam loops")
    np.savez(sys.argv[2], outer=loops[o], n_holes=len(holes), **{f"hole_{i}": h for i, h in enumerate(holes)})


if __name__ == "__main__":
    main()
