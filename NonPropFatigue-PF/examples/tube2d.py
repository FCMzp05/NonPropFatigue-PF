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

CASES = {
    "S1": (0.0, 1, 87.0, 1, 0.0),
    "P1": (102.0, 1, 63.0, 1, 0.0),
    "P2": (80.0, 1, 50.0, 1, 0.0),
    "P3": (86.0, 1, 71.0, 1, 0.0),
    "NP1": (102.0, 1, 63.0, 2, 0.0),
    "R001": (138.4, 1, 0.0, 1, 0.0),
    "R004": (101.5, 1, 62.6, 1, 0.0),
    "R005": (101.5, 1, 62.6, 1, 90.0),
    "R006": (101.5, 1, 62.6, 1, 45.0),
}


def notch_mesh(L, WN, HN, h_fine, h_coarse, boxes):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("vormwald")
    occ = gmsh.model.occ
    r, a = HN/2, WN/2-HN/2
    s = occ.addRectangle(-L/2, -L/2, 0, L, L)
    hole, _ = occ.fuse([(2, occ.addRectangle(-a, -r, 0, 2*a, 2*r))],
                       [(2, occ.addDisk(-a, 0, 0, r, r)), (2, occ.addDisk(a, 0, 0, r, r))])
    occ.cut([(2, s)], hole)
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
    ap.add_argument("--case", choices=tuple(CASES), default="NP1")
    ap.add_argument("--load-scale", type=float, default=1.0, help="multiplies both stress amplitudes")
    ap.add_argument("--L", type=float, default=120.0)
    ap.add_argument("--WN", type=float, default=14.0, help="notch width (mm), 3.5 HN")
    ap.add_argument("--HN", type=float, default=4.0, help="notch height (mm)")
    ap.add_argument("--E", type=float, default=210e3)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--ell", type=float, default=0.4)
    ap.add_argument("--h", type=float, default=0.15)
    ap.add_argument("--h-coarse", type=float, default=6.0)
    ap.add_argument("--Gc", type=float, default=50.0)
    ap.add_argument("--alpha-T", type=float, default=None)
    ap.add_argument("--sigma-e", type=float, default=200.0)
    ap.add_argument("--fat-deg", default="asym")
    ap.add_argument("--kappa", type=float, default=0.5, help="slope of the log fatigue degradation")
    ap.add_argument("--sy0", type=float, default=None, help="cyclic yield stress (MPa); enables J2 plasticity")
    ap.add_argument("--H-kin", type=float, default=0.0, help="linear kinematic hardening modulus (MPa)")
    ap.add_argument("--H-iso", type=float, default=0.0)
    ap.add_argument("--chi-p", type=float, default=1.0, help="weight of the plastic work in the fatigue history")
    ap.add_argument("--substeps", type=int, default=32)
    ap.add_argument("--dN-max", type=int, default=500)
    ap.add_argument("--dd-target", type=float, default=0.02)
    ap.add_argument("--df-target", type=float, default=0.02)
    ap.add_argument("--stagger-tol", type=float, default=1e-4, help="max change of d between staggered iterations")
    ap.add_argument("--s-stop", type=float, default=20.0, help="stop when a crack reaches this distance (mm)")
    ap.add_argument("--mask-cap", type=int, default=0, help="> 0: stagger iterations after which the tension/compression "
                    "mask no longer holds the stagger open")
    ap.add_argument("--vctd-Kc", type=float, default=0.0, help="alpha = Delta K_I/Kc [MPa sqrt(mm)]; 0: no steering")
    ap.add_argument("--vctd-R", type=float, default=5.0)
    ap.add_argument("--vctd-w", type=float, default=3.0)
    ap.add_argument("--notch-steer", action="store_true",
                    help="also steer in a band around each round notch end from the cycle of the jump between "
                         "the notch faces, as at a blunt crack tip; crack tips override it where they overlap")
    ap.add_argument("--tau-sign", type=float, default=1.0, help="-1 reverses the shear history")
    ap.add_argument("--amr-R", type=float, default=8.0)
    ap.add_argument("--amr-d", type=float, default=0.1)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"runs/vorm_{a.case}_Kc{a.vctd_Kc:g}")
    out.mkdir(parents=True, exist_ok=True)

    sa, fs, ta, ft, lag = CASES[a.case]
    sa, ta = sa*a.load_scale, a.tau_sign*ta*a.load_scale
    r_n, a_n = a.HN/2, a.WN/2-a.HN/2

    def notch_dist(p):
        p = np.atleast_2d(p)
        return np.hypot(np.clip(np.abs(p[:, 0])-a_n, 0, None), p[:, 1])-r_n

    def notch_foot(p):
        c = np.array([np.clip(p[0], -a_n, a_n), 0.0])
        v = p-c
        return c+r_n*v/max(np.linalg.norm(v), 1e-12)

    R = a.amr_R*a.ell
    boxes = [(-a.WN/2-R, -a_n+r_n, -r_n-R, r_n+R), (a_n-r_n, a.WN/2+R, -r_n-R, r_n+R)]
    nodes, tris = notch_mesh(a.L, a.WN, a.HN, a.h, a.h_coarse, boxes)
    if a.jitter > 0:
        nodes = jitter(nodes, tris, a.jitter, a.seed)
    tris = amr.label_longest(nodes, tris)
    xc = nodes[tris].mean(1)
    print(f"case {a.case}: sigma_a={sa:g} (f={fs}), tau_a={ta:g} (f={ft}); mesh {len(nodes)} nodes, "
          f"{len(tris)} triangles", flush=True)

    sig = lambda t: sa*np.sin(2*np.pi*fs*np.asarray(t))
    tau = lambda t: ta*np.sin(2*np.pi*ft*np.asarray(t)-math.radians(lag))

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
    mat = Material(E=a.E, nu=a.nu, Gc=a.Gc, ell=a.ell, plane="stress", alpha_T=a.alpha_T, sigma_e=a.sigma_e,
                   fat_deg=a.fat_deg, kappa=a.kappa, sy0=a.sy0, H_kin=a.H_kin, H_iso=a.H_iso, chi_p=a.chi_p)
    ctl = Controls(substeps=a.substeps, n_cycles=10**8, dN_max=a.dN_max, dd_target=a.dd_target,
                   df_target=a.df_target, stagger_tol=a.stagger_tol, mask_cap=a.mask_cap, out=str(out), save_every=10)
    s = FatigueSolver(nodes, tris, mat, ch, fixed, ctl, thickness=1.0)
    np.savez_compressed(out/"mesh.npz", nodes=nodes, tris=tris, thickness=np.ones(len(tris)),
                        notch=np.array([a.WN, a.HN]))
    json.dump(vars(a), open(out/"args.json", "w"), indent=1)

    track: dict[str, dict] = {}
    steer = a.vctd_Kc > 0
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    from vctdpf.criteria import vctd_odd
    if steer:
        s.record_u = True
        s.A_el = torch.eye(2, dtype=torch.float64, device=s.d.device).repeat(len(tris), 1, 1)
    tree = cKDTree(nodes)
    logs: dict[str, object] = {}
    state: dict[str, np.ndarray] = {}
    min_len = 6.0*a.ell

    def tangent(key):
        P = track[key]["path"]
        p = P[-1]
        back = [q for q in P if np.hypot(*(p-q)) >= 1.0]
        v = p-(back[-1] if back else P[0])
        return v/max(np.linalg.norm(v), 1e-12)

    def find_cracks(d):
        on = np.flatnonzero((d > 0.9) & (notch_dist(nodes) > a.ell))
        if len(on) == 0:
            return []
        P = nodes[on]
        pairs = cKDTree(P).query_pairs(2.5*a.h, output_type="ndarray")
        n = len(on)
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else \
            coo_matrix((n, n))
        nc, lab = connected_components(g, directed=False)
        cracks = []
        for c in range(nc):
            Q = P[lab == c]
            dist = notch_dist(Q)
            if dist.max() < 2*a.ell:
                continue
            root = notch_foot(Q[np.argmin(dist)])
            tip = Q[np.argmax(np.hypot(*(Q-root).T))]
            cracks.append((root, tip, Q))
        return cracks

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
        e = np.cos(th)*tdir+np.sin(th)*ndir
        state[key] = e
        rel = xc-tip
        zone = (np.hypot(rel[:, 0], rel[:, 1]) < a.vctd_R*a.ell) & (rel@tdir > -a.ell)
        idx = torch.as_tensor(np.flatnonzero(zone), device=sol.d.device)
        et = torch.as_tensor(e, dtype=torch.float64, device=sol.d.device)
        tt = torch.as_tensor(tdir, dtype=torch.float64, device=sol.d.device)
        dc = a.vctd_Kc*8.0*np.mean(np.sqrt(rbs/(2*np.pi)))/a.E
        al = np.ptp(dI)/dc
        lam = min(al*abs(math.sin(th)), 30.0)
        sh = math.sinh(lam)/lam if lam > 1e-12 else 1.0
        sol.A_el[idx] = math.cosh(lam)*torch.eye(2, dtype=torch.float64, device=sol.d.device) \
            + sh*al*(torch.outer(et, et)-torch.outer(tt, tt))
        if key not in logs:
            logs[key] = open(out/f"vctd_{key}.txt", "a")
        logs[key].write(f"{tip[0]:.4f} {tip[1]:.4f} {math.degrees(math.atan2(tdir[1], tdir[0])):.2f} "
                        f"{math.degrees(th):.2f} {dI.min():.4e} {dI.max():.4e} {dII.min():.4e} {dII.max():.4e} "
                        f"{al:.4f}\n")
        logs[key].flush()
        return math.degrees(th)

    def steer_notch(sol, side):
        key = "nR" if side > 0 else "nL"
        tip = np.array([side*a.WN/2, 0.0])
        tdir = np.array([side, 0.0])
        ndir = np.array([0.0, side])
        U = torch.stack(sol.u_hist).cpu().numpy().reshape(len(sol.u_hist), -1, 2)
        J = 0.0
        rbs = a.HN/2+np.linspace(2.0, 5.0, 7)*a.ell
        for rb in rbs:
            xb = tip-rb*tdir
            iu = tree.query(xb+r_n*ndir, k=4)[1]
            il = tree.query(xb-r_n*ndir, k=4)[1]
            J = J+(U[:, iu].mean(1)-U[:, il].mean(1))/len(rbs)
        dI, dII = J@ndir, J@tdir
        th = vctd_odd(dI, dII, star=False, mu=float("inf"), rule="sum")[0]
        e = np.cos(th)*tdir+np.sin(th)*ndir
        zone = (notch_dist(xc) < a.vctd_R*a.ell) & ((xc[:, 0]-side*a_n)*side > 0)
        idx = torch.as_tensor(np.flatnonzero(zone), device=sol.d.device)
        et = torch.as_tensor(e, dtype=torch.float64, device=sol.d.device)
        tt = torch.as_tensor(tdir, dtype=torch.float64, device=sol.d.device)
        dc = a.vctd_Kc*8.0*np.mean(np.sqrt(rbs/(2*np.pi)))/a.E
        al = np.ptp(dI)/dc
        lam = min(al*abs(math.sin(th)), 30.0)
        sh = math.sinh(lam)/lam if lam > 1e-12 else 1.0
        sol.A_el[idx] = math.cosh(lam)*torch.eye(2, dtype=torch.float64, device=sol.d.device) \
            + sh*al*(torch.outer(et, et)-torch.outer(tt, tt))
        if key not in logs:
            logs[key] = open(out/f"vctd_{key}.txt", "a")
        logs[key].write(f"{math.degrees(th):.2f} {dI.min():.4e} {dI.max():.4e} {dII.min():.4e} {dII.max():.4e} "
                        f"{al:.4f}\n")
        logs[key].flush()
        return math.degrees(th)

    def adapt(sol):
        nonlocal nodes, tris, xc, tree
        pts = [tr["path"][-1] for tr in track.values()]
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
        tree = cKDTree(nodes)
        print(f"  -> refined ({len(mids)} passes): {m0} -> {len(tris)} elements", flush=True)

    def hook(sol):
        d = sol.d.cpu().numpy()
        found = find_cracks(d)
        for root, tip, Q in found:
            key = next((k for k, tr in track.items()
                        if np.hypot(*(Q-tr["path"][-1]).T).min() < 3*a.ell), None)
            if key is None:
                key = f"c{len(track)+1}"
                track[key] = {"root": root, "path": [root.copy()]}
                print(f"  -> new crack {key} at ({root[0]:.2f}, {root[1]:.2f})", flush=True)
            tr = track[key]
            if np.hypot(*(tip-tr["path"][-1])) > 0.5*a.h:
                tr["path"].append(tip.copy())
        th = {}
        if steer and a.notch_steer:
            for side in (1, -1):
                th["nR" if side > 0 else "nL"] = steer_notch(sol, side)
        if steer:
            for k, tr in track.items():
                if notch_dist(tr["path"][-1])[0] >= min_len:
                    th[k] = steer_tip(sol, k)
        adapt(sol)
        far = {k: float(notch_dist(tr["path"][-1])[0]) for k, tr in track.items()}
        edge = max((np.abs(tr["path"][-1]).max() for tr in track.values()), default=0.0)
        if (far and max(far.values()) >= a.s_stop) or edge > a.L/2-3.0:
            sol.ctl.n_cycles = 0
        for k, tr in track.items():
            np.savetxt(out/f"tip_path_{k}.txt", np.array(tr["path"]))
        json.dump({k: {"path": [list(map(float, p)) for p in v["path"]]} for k, v in track.items()},
                  open(out/"track.json", "w"))
        return dict(n_cracks=len(track), s_max=max(far.values(), default=0.0),
                    **{f"s_{k}": v for k, v in far.items()}, **{f"th_{k}": v for k, v in th.items()})

    s.run(hooks=hook)


if __name__ == "__main__":
    main()
