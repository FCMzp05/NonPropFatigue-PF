from __future__ import annotations

import numpy as np
import gmsh


def _extract(dim_tag=2):
    gmsh.model.mesh.generate(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    nodes = xyz.reshape(-1, 3)[:, :2]
    idx = np.full(int(tags.max())+1, -1, dtype=np.int64)
    idx[tags.astype(np.int64)] = np.arange(len(tags))
    tris = []
    for et, _, conn in zip(*gmsh.model.mesh.getElements(2)):
        if et == 2:
            tris.append(idx[conn.astype(np.int64)].reshape(-1, 3))
    tris = np.vstack(tris)
    used = np.unique(tris)
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    nodes, tris = nodes[used], remap[tris]
    x = nodes[tris]
    det = (x[:, 1, 0]-x[:, 0, 0])*(x[:, 2, 1]-x[:, 0, 1])-(x[:, 1, 1]-x[:, 0, 1])*(x[:, 2, 0]-x[:, 0, 0])
    tris[det < 0] = tris[det < 0][:, [0, 2, 1]]
    return nodes, tris


def _band_field(h_fine, h_coarse, box):
    f = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f, "VIn", h_fine)
    gmsh.model.mesh.field.setNumber(f, "VOut", h_coarse)
    for k, v in zip(("XMin", "XMax", "YMin", "YMax"), box):
        gmsh.model.mesh.field.setNumber(f, k, v)
    gmsh.model.mesh.field.setNumber(f, "Thickness", 0.3*(box[1]-box[0]+box[3]-box[2]))
    return f


def slit_polygon(path: np.ndarray, w: float) -> np.ndarray:
    P = np.asarray(path, float)
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.c_[-t[:, 1], t[:, 0]]*(w/2)
    inner = slice(1, -1)
    return np.vstack([P[:1], (P+n)[inner], P[-1:], (P-n)[inner][::-1]])


def _add_loop(geo, pts: np.ndarray) -> int:
    tags = [geo.addPoint(float(x), float(y), 0.0) for x, y in pts]
    lines = [geo.addLine(tags[i], tags[(i+1) % len(tags)]) for i in range(len(tags))]
    return geo.addCurveLoop(lines)


def cruciform(geom_npz: str, cracks: list[np.ndarray], boxes: list[tuple], h_fine: float, h_coarse: float,
              slit_w: float = 0.02, h_mid: float | None = None, mid_box: tuple | None = None):
    g = np.load(geom_npz)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cruciform")
    geo = gmsh.model.geo
    outer = _add_loop(geo, g["outer"])
    holes = [_add_loop(geo, g[f"hole_{i}"]) for i in range(int(g["n_holes"]))]
    holes += [_add_loop(geo, slit_polygon(c, slit_w)) for c in cracks]
    geo.addPlaneSurface([outer]+holes)
    geo.synchronize()
    fields = [_band_field(h_fine, h_coarse, b) for b in boxes]
    if mid_box is not None:
        fields.append(_band_field(h_mid, h_coarse, mid_box))
    tip = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(tip, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(tip)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_coarse)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    nodes, tris = _extract()
    gmsh.finalize()
    return nodes, tris


def sent(L=1.0, a=0.5, notch_w=0.0, h_fine=0.0025, h_coarse=0.05, band=(0.45, 1.0, 0.44, 0.56)):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("sent")
    occ = gmsh.model.occ
    s = occ.addRectangle(0, 0, 0, L, L)
    w = max(notch_w, 1e-4*L)
    n = occ.addRectangle(0, L/2-w/2, 0, a, w)
    occ.cut([(2, s)], [(2, n)])
    occ.synchronize()
    f = _band_field(h_fine, h_coarse, band)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_coarse)
    nodes, tris = _extract()
    gmsh.finalize()
    return nodes, tris
