from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np


def read_inp(path: Path) -> dict:
    lines = Path(path).read_text().splitlines()
    nodes, elems, nsets, elsets, sections = {}, {}, {}, {}, {}
    i, mode, cur, gen, etype = 0, None, None, False, None
    in_instance = False
    while i < len(lines):
        s = lines[i].strip()
        i += 1
        if not s or s.startswith("**"):
            continue
        if s.startswith("*"):
            kw = s.lower()
            mode = None
            if kw.startswith("*instance"):
                in_instance = True
            elif kw.startswith("*end instance"):
                in_instance = False
            elif kw.startswith("*node") and not kw.startswith("*node output") and in_instance:
                mode = "node"
            elif kw.startswith("*element,") and in_instance:
                mode = "elem"
                etype = re.search(r"type=(\w+)", s, re.I).group(1)
            elif kw.startswith("*nset") and in_instance:
                mode, cur = "nset", re.search(r"nset=([^,]+)", s, re.I).group(1)
                gen = "generate" in kw
                nsets[cur] = []
            elif kw.startswith("*elset") and in_instance:
                mode, cur = "elset", re.search(r"elset=([^,]+)", s, re.I).group(1)
                gen = "generate" in kw
                elsets[cur] = []
            elif kw.startswith("*solid section"):
                es = re.search(r"elset=([^,]+)", s, re.I).group(1)
                t = lines[i].split(",")[0].strip()
                sections[es] = float(t) if t else 1.0
            continue
        vals = [v for v in s.split(",") if v.strip()]
        if mode == "node":
            nodes[int(vals[0])] = [float(vals[1]), float(vals[2])]
        elif mode == "elem":
            elems[int(vals[0])] = [int(v) for v in vals[1:]]
        elif mode in ("nset", "elset"):
            ints = [int(v) for v in vals]
            tgt = nsets if mode == "nset" else elsets
            if gen:
                a, b, st = ints[0], ints[1], ints[2] if len(ints) > 2 else 1
                tgt[cur].extend(range(a, b+1, st))
            else:
                tgt[cur].extend(ints)
    nid = np.array(sorted(nodes))
    X = np.array([nodes[k] for k in nid])
    eid = np.array(sorted(elems))
    C = np.array([elems[k] for k in eid])
    return dict(node_ids=nid, X=X, elem_ids=eid, conn=C, etype=etype, nsets=nsets, elsets=elsets,
                sections=sections)


def boundary_edges(conn: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    q = conn[:, :4]
    e = np.concatenate([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]], q[:, [3, 0]]])
    key = np.sort(e, axis=1)
    u, cnt = np.unique(key, axis=0, return_counts=True)
    return u[cnt == 1]
