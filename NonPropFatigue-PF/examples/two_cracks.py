from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vctdpf.solver import Channel, Controls, FatigueSolver, Material
from vctdpf import amr
from vctdpf.mesh import _band_field, _extract
from biaxial import edge_forces, jitter


def patch_mesh(L, cracks, slit_w, h_fine, h_coarse, boxes):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("twocrack")
    occ = gmsh.model.occ
    s = occ.addRectangle(-L/2, -L/2, 0, L, L)
    cuts = [(2, occ.addRectangle(p[0], p[1]-slit_w/2, 0, q[0]-p[0], slit_w)) for p, q in cracks]
    occ.cut([(2, s)], cuts)
    occ.synchronize()
    f = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f, "FieldsList", [_band_field(h_fine, h_coarse, b) for b in boxes])
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_coarse)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    nodes, tris = _extract()
    gmsh.finalize()
    return nodes, tris


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=40.0)
    ap.add_argument("--a0", type=float, default=3.0, help="half-length of each crack (mm)")
    ap.add_argument("--c", type=float, default=2.5, help="horizontal offset of the crack centres from x = 0 (mm)")
    ap.add_argument("--sep", type=float, default=2.0, help="vertical distance between the crack planes (mm)")
    ap.add_argument("--sig-c", type=float, default=120.0)
    ap.add_argument("--sig-b", type=float, default=100.0)
    ap.add_argument("--tau-t", type=float, default=60.0)
    ap.add_argument("--phi", type=float, default=90.0)
    ap.add_argument("--E", type=float, default=114e3)
    ap.add_argument("--nu", type=float, default=0.34)
    ap.add_argument("--ell", type=float, default=0.4)
    ap.add_argument("--h", type=float, default=0.15)
    ap.add_argument("--h-coarse", type=float, default=2.0)
    ap.add_argument("--slit-w", type=float, default=0.02)
    ap.add_argument("--Gc", type=float, default=50.0)
    ap.add_argument("--alpha-T", type=float, default=None)
    ap.add_argument("--substeps", type=int, default=16)
    ap.add_argument("--dN-max", type=int, default=500)
    ap.add_argument("--dd-target", type=float, default=0.02)
    ap.add_argument("--df-target", type=float, default=0.02)
    ap.add_argument("--s-stop", type=float, default=8.0, help="stop when an outer tip has grown this far (mm)")
    ap.add_argument("--vctd-odd", action="store_true")
    ap.add_argument("--vctd-Kc", type=float, default=0.0, help="alpha = Delta K_I/Kc [MPa sqrt(mm)]")
    ap.add_argument("--vctd-R", type=float, default=5.0)
    ap.add_argument("--vctd-w", type=float, default=3.0)
    ap.add_argument("--amr-R", type=float, default=8.0)
    ap.add_argument("--amr-d", type=float, default=0.1)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"runs/twocrack_phi{a.phi:g}_Kc{a.vctd_Kc:g}")
    out.mkdir(parents=True, exist_ok=True)

    y1, y2 = a.sep/2, -a.sep/2
    cracks = [(np.array([-a.c-a.a0, y1]), np.array([-a.c+a.a0, y1])),
              (np.array([a.c-a.a0, y2]), np.array([a.c+a.a0, y2]))]
    tips = {"L1": (0, cracks[0][0], np.array([-1.0, 0.0])), "R1": (0, cracks[0][1], np.array([1.0, 0.0])),
            "L2": (1, cracks[1][0], np.array([-1.0, 0.0])), "R2": (1, cracks[1][1], np.array([1.0, 0.0]))}
    outer = ("L1", "R2")
    R = a.amr_R*a.ell
    boxes = [(p[0]-R, p[0]+R, p[1]-R, p[1]+R) for _, p, _ in tips.values()]
    nodes, tris = patch_mesh(a.L, cracks, a.slit_w, a.h, a.h_coarse, boxes)
    if a.jitter > 0:
        nodes = jitter(nodes, tris, a.jitter, a.seed)
    tris = amr.label_longest(nodes, tris)
    xc = nodes[tris].mean(1)
    print(f"mesh: {len(nodes)} nodes, {len(tris)} triangles", flush=True)

    sig = lambda t: a.sig_c+a.sig_b*np.sin(2*np.pi*np.asarray(t))
    tau = lambda t: a.tau_t*np.sin(2*np.pi*np.asarray(t)-np.radians(a.phi))

    def boundary(nodes):
        tol, h2 = 1e-6, a.L/2
        side = {k: np.flatnonzero(m) for k, m in (("b", nodes[:, 1] < -h2+tol), ("t", nodes[:, 1] > h2-tol),
                                                  ("l", nodes[:, 0] < -h2+tol), ("r", nodes[:, 0] > h2-tol))}
        (ib, wb), (it, wt) = (edge_forces(nodes, side[k], 1) for k in "bt")
        (il, wl), (ir, wr) = (edge_forces(nodes, side[k], 0) for k in "lr")
        ch = [Channel("force", np.r_[2*it+1, 2*ib+1], a.L*np.r_[wt, -wb], sig, "sigma"),
              Channel("force", np.r_[2*it, 2*ib, 2*ir+1, 2*il+1], a.L*np.r_[wt, -wb, wr, -wl], tau, "tau")]
        c0 = np.argmin(np.hypot(nodes[:, 0]+h2, nodes[:, 1]+h2))
        c1 = np.argmin(np.hypot(nodes[:, 0]-h2, nodes[:, 1]+h2))
        return ch, np.array([2*c0, 2*c0+1, 2*c1+1])

    ch, fixed = boundary(nodes)
    mat = Material(E=a.E, nu=a.nu, Gc=a.Gc, ell=a.ell, plane="stress", alpha_T=a.alpha_T)
    ctl = Controls(substeps=a.substeps, n_cycles=10**7, dN_max=a.dN_max, dd_target=a.dd_target,
                   df_target=a.df_target, out=str(out), save_every=10)
    s = FatigueSolver(nodes, tris, mat, ch, fixed, ctl, thickness=1.0)
    np.savez_compressed(out/"mesh.npz", nodes=nodes, tris=tris, thickness=np.ones(len(tris)),
                        crack0=np.array([cracks[0][0], cracks[0][1], cracks[1][0], cracks[1][1]]))
    json.dump(vars(a), open(out/"args.json", "w"), indent=1)

    track = {k: {"path": [p.copy()], "t0": t0, "s": 0.0, "live": True} for k, (_, p, t0) in tips.items()}
    steer = a.vctd_Kc > 0
    if steer:
        import torch
        from scipy.spatial import cKDTree
        from vctdpf.criteria import vctd_odd
        s.record_u = True
        tree = cKDTree(nodes)
        s.A_el = torch.eye(2, dtype=torch.float64, device=s.d.device).repeat(len(tris), 1, 1)
        logs = {k: open(out/f"vctd_{k}.txt", "a") for k in tips}
    state = {}

    def tangent(key):
        P = track[key]["path"]
        p = P[-1]
        back = [q for q in P if np.hypot(*(p-q)) >= 1.0]
        if not back:
            return track[key]["t0"]
        v = p-back[-1]
        return v/np.linalg.norm(v)

    def other_crack_points(key):
        ic = tips[key][0]
        pts = [np.linspace(*cracks[1-ic], 40)]
        pts += [np.array(track[k]["path"]) for k in tips if tips[k][0] != ic]
        return np.vstack(pts)

    def steer_tip(sol, key):
        tip = track[key]["path"][-1]
        tdir = tangent(key)
        ndir = np.array([-tdir[1], tdir[0]])
        U = torch.stack(sol.u_hist).cpu().numpy().reshape(len(sol.u_hist), -1, 2)
        w = a.vctd_w*a.ell
        J = 0.0
        rbs = np.linspace(2.0, 5.0, 7)*a.ell
        for rb in rbs:
            xb = tip-rb*tdir
            iu = tree.query(xb+w*ndir, k=4)[1]
            il = tree.query(xb-w*ndir, k=4)[1]
            J = J+(U[:, iu].mean(1)-U[:, il].mean(1))/len(rbs)
        dI, dII = J@ndir, J@tdir
        th = vctd_odd(dI, dII, star=False, mu=float("inf"), rule="sum")[0]
        e = state.get(key, tdir) if th is None else np.cos(th)*tdir+np.sin(th)*ndir
        state[key] = e
        rel = xc-tip
        zone = (np.hypot(rel[:, 0], rel[:, 1]) < a.vctd_R*a.ell) & (rel@tdir > -a.ell)
        idx = torch.as_tensor(np.flatnonzero(zone), device=sol.d.device)
        et = torch.as_tensor(e, dtype=torch.float64, device=sol.d.device)
        tt = torch.as_tensor(tdir, dtype=torch.float64, device=sol.d.device)
        dc = a.vctd_Kc*8.0*np.mean(np.sqrt(rbs/(2*np.pi)))/a.E
        al = np.ptp(dI)/dc
        th_r = math.atan2(e@ndir, e@tdir)
        lam = min(al*abs(math.sin(th_r)), 30.0)
        sh = math.sinh(lam)/lam if lam > 1e-12 else 1.0
        sol.A_el[idx] = math.cosh(lam)*torch.eye(2, dtype=torch.float64, device=sol.d.device) \
            + sh*al*(torch.outer(et, et)-torch.outer(tt, tt))
        logs[key].write(f"{tip[0]:.4f} {tip[1]:.4f} {math.degrees(math.atan2(tdir[1], tdir[0])):.2f} "
                        f"{math.degrees(th_r):.2f} {dI.min():.4e} {dI.max():.4e} {dII.min():.4e} {dII.max():.4e} "
                        f"{al:.4f}\n")
        logs[key].flush()
        return math.degrees(th_r)

    def adapt(sol):
        nonlocal nodes, tris, xc, tree
        pts = [track[k]["path"][-1] for k in tips if track[k]["live"]]
        dn = sol.d.cpu().numpy()

        def marked(r):
            near = np.logical_or.reduce([np.hypot(*(xc-p).T) < r for p in pts]) if pts else np.zeros(len(tris), bool)
            return (near | (dn[tris].max(1) >= a.amr_d)) & (amr.size(nodes, tris) > a.h)

        if not marked(0.5*R).any():
            return
        m0, mids, parent = len(tris), [], np.arange(len(tris))
        while True:
            mark = marked(R)
            if not mark.any():
                break
            nodes, tris, p, ea, eb = amr.refine(nodes, tris, mark)
            dn = np.r_[dn, 0.5*(dn[ea]+dn[eb])]
            xc = nodes[tris].mean(1)
            parent = parent[p]
            mids.append((ea, eb))
        sol.remesh(nodes, tris, parent, mids, *boundary(nodes))
        if steer:
            tree = cKDTree(nodes)
        print(f"  -> refined ({len(mids)} passes): {m0} -> {len(tris)} elements", flush=True)

    def hook(sol):
        d = sol.d.cpu().numpy()
        on = d > 0.9
        P = nodes[on]
        for k in tips:
            tr = track[k]
            if not tr["live"]:
                continue
            p, t = tr["path"][-1], tangent(k)
            rel = P-p
            cand = (np.hypot(rel[:, 0], rel[:, 1]) < 4*a.ell) & (rel@t > 0.5*a.h)
            if cand.any():
                q = P[cand][np.argmax(rel[cand]@t)]
                tr["s"] += float(np.hypot(*(q-p)))
                tr["path"].append(q)
                oc = other_crack_points(k)
                if np.hypot(*(oc-q).T).min() < 2*a.ell:
                    tr["live"] = False
                    print(f"  -> tip {k} linked at ({q[0]:.2f}, {q[1]:.2f})", flush=True)
        th = {}
        if steer:
            for k in tips:
                if track[k]["live"]:
                    th[k] = steer_tip(sol, k)
        adapt(sol)
        so = max(track[k]["s"] for k in outer)
        far = max(np.abs(track[k]["path"][-1]).max() for k in tips)
        if so >= a.s_stop or far > a.L/2-3.0:
            sol.ctl.n_cycles = 0
        for k in tips:
            np.savetxt(out/f"tip_path_{k}.txt", np.array(track[k]["path"]))
        json.dump({k: {"path": [list(map(float, p)) for p in v["path"]], "s": v["s"], "live": v["live"]}
                   for k, v in track.items()}, open(out/"track.json", "w"))
        return dict(s_outer=so, **{f"s_{k}": track[k]["s"] for k in tips}, **{f"th_{k}": v for k, v in th.items()})

    s.run(hooks=hook)


if __name__ == "__main__":
    main()
