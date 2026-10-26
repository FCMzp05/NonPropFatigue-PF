from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vctdpf import amr3d
from vctdpf.solver import Channel, Controls, Material
from vctdpf.solver3d import FatigueSolver3D
from tube2d import CASES

_S, _T = 56.9e3/222.96, 554e3/3114.6
HS_CASES = {
    "N01": (0.8*_S, 0.2*_S, 0.8*_T, 0.2*_T, 0.0),
    "N03": (_S, 0.0, 0.8*_T, 0.2*_T, 0.0),
    "N04": (_S, 0.0, 0.0, _T, 0.0),
    "N05": (0.8*_S, 0.2*_S, _T, 0.0, 0.0),
    "N06": (_S, 0.0, 0.5*_T, 0.5*_T, 0.0),
}


def tube_mesh(Ro, t, Ly, xn, WN_half_r, HN, h_fine, h_coarse, box, extra=(), size_cb=None):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("tube3d")
    occ = gmsh.model.occ
    Ri = Ro-t
    outer = occ.addCylinder(0, -Ly/2, 0, 0, Ly, 0, Ro)
    inner = occ.addCylinder(0, -Ly/2, 0, 0, Ly, 0, Ri)
    tube, _ = occ.cut([(3, outer)], [(3, inner)])
    r_n, z0, dz = HN/2, 0.5*Ri, Ro-0.5*Ri+1.0
    prism, _ = occ.fuse([(3, occ.addBox(-xn, -r_n, z0, 2*xn, HN, dz))],
                        [(3, occ.addCylinder(-xn, 0, z0, 0, 0, dz, r_n)),
                         (3, occ.addCylinder(xn, 0, z0, 0, 0, dz, r_n))])
    occ.cut(tube, prism)
    occ.synchronize()
    fields = []
    for h_z, b in ((h_fine, box), *extra):
        f = gmsh.model.mesh.field.add("Box")
        for k, v in zip(("VIn", "VOut", "XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax", "Thickness"),
                        (h_z, h_coarse, *b, 4.0)):
            gmsh.model.mesh.field.setNumber(f, k, v)
        fields.append(f)
    fmin = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
    if size_cb is not None:
        gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: min(lc, size_cb(x, y, z)))
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_coarse)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.model.mesh.generate(3)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    nodes = xyz.reshape(-1, 3)
    idx = np.full(int(tags.max())+1, -1, dtype=np.int64)
    idx[tags.astype(np.int64)] = np.arange(len(tags))
    tets = []
    for et, _, conn in zip(*gmsh.model.mesh.getElements(3)):
        if et == 4:
            tets.append(idx[conn.astype(np.int64)].reshape(-1, 4))
    gmsh.finalize()
    tets = np.vstack(tets)
    used = np.unique(tets)
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    nodes, tets = nodes[used], remap[tets]
    x = nodes[tets]
    det = np.linalg.det(np.stack([x[:, 1]-x[:, 0], x[:, 2]-x[:, 0], x[:, 3]-x[:, 0]], axis=2))
    tets[det < 0] = tets[det < 0][:, [0, 2, 1, 3]]
    return nodes, tets


def face_weights(nodes, tets, on):
    w = np.zeros(len(nodes))
    for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        F = tets[:, f]
        sel = on[F].all(1)
        P = nodes[F[sel]]
        A = 0.5*np.linalg.norm(np.cross(P[:, 1]-P[:, 0], P[:, 2]-P[:, 0]), axis=1)
        np.add.at(w, F[sel].ravel(), np.repeat(A/3, 3))
    i = np.flatnonzero(w > 0)
    return i, w[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=tuple(CASES)+tuple(HS_CASES), default="NP1")
    ap.add_argument("--precrack", type=float, default=0.0, help="half arc length (mm) of an initial through crack "
                    "along y = 0 beyond the notch, imposed through the history field")
    ap.add_argument("--load-scale", type=float, default=1.0)
    ap.add_argument("--tau-sign", type=float, default=1.0)
    ap.add_argument("--tau-linear", action="store_true", help="end-face shear stress tau r/Ro (elastic torsion, "
                    "nominal tau at the outer radius as in Hos & Vormwald 2016) instead of uniform tau")
    ap.add_argument("--Ro", type=float, default=21.2, help="outer radius (mm), Hos & Vormwald (2016) Fig. 1")
    ap.add_argument("--t", type=float, default=2.6, help="wall thickness (mm)")
    ap.add_argument("--Ly", type=float, default=146.0, help="free length between the grips (mm)")
    ap.add_argument("--WN", type=float, default=14.0, help="notch width along the outer circumference (mm)")
    ap.add_argument("--HN", type=float, default=4.0)
    ap.add_argument("--E", type=float, default=210e3)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--ell", type=float, default=0.7)
    ap.add_argument("--h", type=float, default=0.35)
    ap.add_argument("--h-coarse", type=float, default=3.0)
    ap.add_argument("--fine-s", type=float, default=22.0, help="half arc length of the refined region (mm)")
    ap.add_argument("--fine-y", type=float, default=10.0, help="half axial extent of the refined region (mm)")
    ap.add_argument("--Gc", type=float, default=50.0)
    ap.add_argument("--alpha-T", type=float, default=None)
    ap.add_argument("--sigma-e", type=float, default=200.0)
    ap.add_argument("--fat-deg", default="asym")
    ap.add_argument("--substeps", type=int, default=24)
    ap.add_argument("--dN-max", type=int, default=500)
    ap.add_argument("--dd-target", type=float, default=0.02)
    ap.add_argument("--df-target", type=float, default=0.02)
    ap.add_argument("--stagger-tol", type=float, default=1e-4, help="max change of d between staggered iterations")
    ap.add_argument("--pcg-tol", type=float, default=0.0, help="> 0: displacement by CG preconditioned with the last "
                    "factorization, to this relative residual")
    ap.add_argument("--s-stop", type=float, default=14.0, help="stop when a crack reaches this distance (mm)")
    ap.add_argument("--mask-cap", type=int, default=0, help="> 0: stagger iterations after which the tension/compression "
                    "mask no longer holds the stagger open")
    ap.add_argument("--n-slice", type=int, default=3)
    ap.add_argument("--vctd-Kc", type=float, default=0.0)
    ap.add_argument("--vctd-R", type=float, default=5.0)
    ap.add_argument("--vctd-w", type=float, default=3.0)
    ap.add_argument("--notch-steer", action="store_true")
    ap.add_argument("--tan-fit", type=float, default=0.0,
                    help="tip tangent from a line fit to the path within this distance of the tip (mm); 0: chord")
    ap.add_argument("--amr", action="store_true", help="start with h-mid in the wall around the notch, h at the "
                    "notch ends, and refine to h where d >= amr-d or within amr-R ell of a crack tip")
    ap.add_argument("--h-mid", type=float, default=1.0)
    ap.add_argument("--amr-R", type=float, default=4.0)
    ap.add_argument("--amr-d", type=float, default=0.1)
    ap.add_argument("--coarsen-R", type=float, default=0.0, help="> 0: remesh from scratch whenever the mesh has grown "
                    "by --coarsen-grow, keeping h on the damage band within coarsen-R ell of a crack tip and h-wake "
                    "on the band further back")
    ap.add_argument("--h-wake", type=float, default=None, help="element size on the crack wake (default 2 h)")
    ap.add_argument("--coarsen-grow", type=float, default=1.3)
    ap.add_argument("--mesh-only", action="store_true")
    ap.add_argument("--restart", default=None, help="snapshot to continue from (same --out and options)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"runs/tube_{a.case}_Kc{a.vctd_Kc:g}")
    out.mkdir(parents=True, exist_ok=True)

    if a.case in HS_CASES:
        sm, sa, tm, ta, lag = HS_CASES[a.case]
        fs = ft = 1
    else:
        sa, fs, ta, ft, lag = CASES[a.case]
        sm = tm = 0.0
    sm, sa = sm*a.load_scale, sa*a.load_scale
    tm, ta = a.tau_sign*tm*a.load_scale, a.tau_sign*ta*a.load_scale
    Ro, Ri = a.Ro, a.Ro-a.t
    r_n = a.HN/2
    xn = Ro*math.sin((a.WN/2-r_n)/Ro)
    ph_f = (a.fine_s)/Ro
    box = (-Ro*math.sin(min(ph_f, math.pi/2)), Ro*math.sin(min(ph_f, math.pi/2)), -a.fine_y, a.fine_y,
           Ri*math.cos(min(ph_f, math.pi/2))-0.5, Ro+0.5)
    R_amr = a.amr_R*a.ell
    if a.amr:
        zb = (Ri*math.cos(min(ph_f, math.pi/2))-0.5, Ro+0.5)
        xe = max(xn+r_n, Ro*math.sin(a.precrack/Ro))
        ends = [(a.h, (xn-r_n, xe+R_amr, -r_n-R_amr, r_n+R_amr, *zb)),
                (a.h, (-xe-R_amr, -xn+r_n, -r_n-R_amr, r_n+R_amr, *zb))]
        nodes, tets = tube_mesh(Ro, a.t, a.Ly, xn, None, a.HN, a.h_mid, a.h_coarse, box, extra=ends)
    else:
        nodes, tets = tube_mesh(Ro, a.t, a.Ly, xn, None, a.HN, a.h, a.h_coarse, box)
    c0 = nodes[tets].mean(1)
    at_end = (np.abs(np.abs(c0[:, 0])-xn) < r_n) & (np.abs(c0[:, 1]) < r_n)
    h_ref = 1.1*float(np.median(amr3d.size(nodes, tets[at_end])))
    print(f"case {a.case}: sigma={sm:g}+{sa:g} sin (f={fs}), tau={tm:g}+{ta:g} sin (f={ft}, lag {lag:g}); mesh {len(nodes)} nodes, "
          f"{len(tets)} tetrahedra; refine above size {h_ref:.3f}", flush=True)
    if a.mesh_only:
        np.savez_compressed(out/"mesh.npz", nodes=nodes, tets=tets)
        return
    if a.restart == "latest":
        a.restart = str(max(out.glob("snap_*.npz"), key=lambda p: int(p.stem[5:])))
    if a.restart:
        z = np.load(a.restart)
        if "nodes" in z.files:
            nodes, tets = z["nodes"], z["tris"]
        print(f"restart mesh {len(nodes)} nodes, {len(tets)} tetrahedra", flush=True)

    def geometry(nodes, tets):
        xc = nodes[tets].mean(1)
        phi_c = np.arctan2(xc[:, 0], xc[:, 2])
        return (xc, np.arctan2(nodes[:, 0], nodes[:, 2]), np.hypot(nodes[:, 0], nodes[:, 2]), phi_c,
                np.hypot(xc[:, 0], xc[:, 2]), np.c_[Ro*phi_c, xc[:, 1]])

    xc, phi_n, r_nod, phi_c, r_c, uc = geometry(nodes, tets)

    def notch_dist(p3):
        p3 = np.atleast_2d(p3)
        return np.hypot(np.clip(np.abs(p3[:, 0])-xn, 0, None), p3[:, 1])-r_n

    sig = lambda t: sm+sa*np.sin(2*np.pi*fs*np.asarray(t))
    tau = lambda t: tm+ta*np.sin(2*np.pi*ft*np.asarray(t)-math.radians(lag))

    def boundary(nodes, tets):
        top = np.abs(nodes[:, 1]-a.Ly/2) < 1e-6
        bot = np.abs(nodes[:, 1]+a.Ly/2) < 1e-6
        it_, wt = face_weights(nodes, tets, top)
        ph = np.arctan2(nodes[it_, 0], nodes[it_, 2])
        wtau = wt*np.hypot(nodes[it_, 0], nodes[it_, 2])/Ro if a.tau_linear else wt
        ch = [Channel("force", 3*it_+1, wt, sig, "sigma"),
              Channel("force", np.r_[3*it_, 3*it_+2], np.r_[wtau*np.cos(ph), -wtau*np.sin(ph)], tau, "tau")]
        ib = np.flatnonzero(bot)
        return ch, np.r_[3*ib, 3*ib+1, 3*ib+2]

    ch, fixed = boundary(nodes, tets)
    mat = Material(E=a.E, nu=a.nu, Gc=a.Gc, ell=a.ell, plane="strain", alpha_T=a.alpha_T, sigma_e=a.sigma_e,
                   fat_deg=a.fat_deg)
    ctl = Controls(substeps=a.substeps, n_cycles=10**8, dN_max=a.dN_max, dd_target=a.dd_target,
                   df_target=a.df_target, stagger_tol=a.stagger_tol, pcg_tol=a.pcg_tol, mask_cap=a.mask_cap,
                   out=str(out), save_every=10)
    s = FatigueSolver3D(nodes, tets, mat, ch, fixed, ctl)
    print(f"setup {s.t_setup:.1f}s, {3*len(nodes)} dofs", flush=True)
    if a.restart:
        s.n_remesh = 1
    else:
        if a.precrack > 0:
            pc = (np.abs(uc[:, 1]) < 0.75*a.h) & (np.abs(uc[:, 0]) <= a.precrack)
            import torch
            s.H[torch.as_tensor(np.flatnonzero(pc), device=s.H.device)] = 500.0*a.Gc/a.ell
            print(f"pre-crack: {int(pc.sum())} elements, |s| <= {a.precrack:g} mm", flush=True)
        np.savez_compressed(out/"mesh.npz", nodes=nodes, tets=tets, notch=np.array([a.WN, a.HN, xn, Ro, a.t]))
        json.dump(vars(a), open(out/"args.json", "w"), indent=1)

    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    from vctdpf.criteria import vctd_odd
    steer = a.vctd_Kc > 0
    rs = np.linspace(Ri, Ro, a.n_slice)
    dr = a.t/(a.n_slice-1)
    slab = np.argmin(np.abs(r_c[:, None]-rs[None, :]), axis=1)
    surf_tol = 0.03
    surf = r_nod > Ro-surf_tol
    tree = cKDTree(nodes)
    if steer:
        s.record_u = True
        s.A_el = torch.eye(3, dtype=torch.float64, device=s.d.device).repeat(len(tets), 1, 1)
    track: dict[str, dict] = {}
    if a.restart and (out/"track.json").exists():
        for k, v in json.load(open(out/"track.json")).items():
            P = [np.array(p) for p in v["path"]]
            track[k] = {"root": P[0], "path": P, "far": v["far"]}
    logs: dict[str, object] = {}
    min_len = 6.0*a.ell
    coarsen = a.amr and a.coarsen_R > 0
    R_keep = a.coarsen_R*a.ell
    h_wake = a.h_wake or 2.0*a.h
    hw_ref = h_ref*h_wake/a.h
    n_base = len(tets)

    def to3(su, y, r):
        p = su/Ro
        return np.array([r*math.sin(p), y, r*math.cos(p)])

    def frame(su):
        p = su/Ro
        return np.array([math.cos(p), 0.0, -math.sin(p)]), np.array([0.0, 1.0, 0.0]), \
            np.array([math.sin(p), 0.0, math.cos(p)])

    def apply(sol, key, tip_u, tdir, U, zone_fn, rbs, off):
        ndir = np.array([-tdir[1], tdir[0]])
        es, ey, er = frame(tip_u[0])
        th_all = []
        for k, r in enumerate(rs):
            J = 0.0
            for rb in rbs:
                xb = tip_u-rb*tdir
                iu = tree.query(to3(*(xb+off*ndir), r), k=6)[1]
                il = tree.query(to3(*(xb-off*ndir), r), k=6)[1]
                J = J+(U[:, iu].mean(1)-U[:, il].mean(1))/len(rbs)
            Ju = np.c_[J@es, J@ey]
            dI, dII = Ju@ndir, Ju@tdir
            th = vctd_odd(dI, dII, star=False, mu=float("inf"), rule="sum")[0]
            e2 = np.cos(th)*tdir+np.sin(th)*ndir
            e3 = e2[0]*es+e2[1]*ey
            t3 = tdir[0]*es+tdir[1]*ey
            idx = torch.as_tensor(np.flatnonzero(zone_fn() & (slab == k)), device=sol.d.device)
            dc = a.vctd_Kc*8.0*np.mean(np.sqrt(rbs/(2*np.pi)))/a.E
            al = np.ptp(dI)/dc
            lam = min(al*abs(math.sin(th)), 30.0)
            sh = math.sinh(lam)/lam if lam > 1e-12 else 1.0
            T = lambda v: torch.as_tensor(v, dtype=torch.float64, device=sol.d.device)
            Pr = torch.outer(T(er), T(er))
            I3 = torch.eye(3, dtype=torch.float64, device=sol.d.device)
            sol.A_el[idx] = math.cosh(lam)*(I3-Pr)+Pr+sh*al*(torch.outer(T(e3), T(e3))-torch.outer(T(t3), T(t3)))
            if key not in logs:
                logs[key] = open(out/f"vctd_{key}.txt", "a")
            logs[key].write(f"{k} {tip_u[0]:.4f} {tip_u[1]:.4f} {math.degrees(math.atan2(tdir[1], tdir[0])):.2f} "
                            f"{math.degrees(th):.2f} {dI.min():.4e} {dI.max():.4e} {dII.min():.4e} {dII.max():.4e} "
                            f"{al:.4f}\n")
            logs[key].flush()
            th_all.append(math.degrees(th))
        return float(np.mean(th_all))

    def steer_notch(sol, side, U):
        su_end = side*(a.WN/2)
        tdir = np.array([float(side), 0.0])
        rbs = r_n+np.linspace(2.0, 5.0, 7)*a.ell
        zone = lambda: (notch_dist(xc) < a.vctd_R*a.ell) & ((xc[:, 0]-side*xn)*side > 0) & (xc[:, 2] > 0)
        return apply(sol, "nR" if side > 0 else "nL", np.array([su_end, 0.0]), tdir, U, zone, rbs, r_n)

    def tangent(key):
        P = track[key]["path"]
        p = P[-1]
        if a.tan_fit > 0:
            Q = np.array(P)
            W = Q[np.hypot(*(Q-p).T) <= a.tan_fit]
            ch = p-track[key]["root"]
            ch = ch/max(np.linalg.norm(ch), 1e-12)
            if len(W) >= 3:
                v = np.linalg.svd(W-W.mean(0))[2][0]
                v = v if v@ch >= 0 else -v
                if v@ch >= 0.5:
                    return v
            return ch
        back = [q for q in P if np.hypot(*(p-q)) >= 1.0]
        v = p-(back[-1] if back else P[0])
        return v/max(np.linalg.norm(v), 1e-12)

    def steer_tip(sol, key, U):
        tip = track[key]["path"][-1]
        tdir = tangent(key)
        rbs = np.linspace(2.0, 5.0, 7)*a.ell
        rel = lambda: uc-tip
        zone = lambda: (np.hypot(*rel().T) < a.vctd_R*a.ell) & (rel()@tdir > -a.ell)
        return apply(sol, key, tip, tdir, U, zone, rbs, a.vctd_w*a.ell)

    def find_cracks(d):
        on = np.flatnonzero((d > 0.9) & surf & (notch_dist(nodes) > a.ell) & (nodes[:, 2] > 0))
        if len(on) == 0:
            return []
        P = np.c_[Ro*phi_n[on], nodes[on, 1]]
        pairs = cKDTree(P).query_pairs(2.5*a.h, output_type="ndarray")
        n = len(on)
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else \
            coo_matrix((n, n))
        nc, lab = connected_components(g, directed=False)
        cracks = []
        for c in range(nc):
            Q, Q3 = P[lab == c], nodes[on][lab == c]
            dist = notch_dist(Q3)
            if dist.max() < 2*a.ell:
                continue
            root = Q[np.argmin(dist)]
            tip = Q[np.argmax(np.hypot(*(Q-root).T))]
            cracks.append((root, tip, Q, float(dist.max())))
        return cracks

    def near_tips(p2, r, pts):
        return np.logical_or.reduce([np.hypot(*(p2-p).T) < r for p in pts]) if pts else np.zeros(len(p2), bool)

    def rebuild(sol, pts, dn):
        nonlocal nodes, tets, xc, phi_n, r_nod, phi_c, r_c, uc, slab, surf, tree, n_base
        from scipy.spatial import cKDTree as KD
        band = KD(nodes[dn >= 0.3*a.amr_d])
        P = np.array(pts)

        def cb(x, y, z):
            dt = np.hypot(P[:, 0]-Ro*math.atan2(x, z), P[:, 1]-y).min()
            if dt < R_amr:
                return 0.85*a.h
            if band.query((x, y, z), distance_upper_bound=h_wake)[0] < h_wake:
                return 0.85*a.h if dt < R_keep else h_wake
            return 1e9

        m0 = len(tets)
        n1, t1 = tube_mesh(Ro, a.t, a.Ly, xn, None, a.HN, a.h_mid, a.h_coarse, box, size_cb=cb)
        ei, w = amr3d.locate(nodes, tets, n1)
        ec = amr3d.locate(nodes, tets, n1[t1].mean(1))[0]
        wake = ~near_tips(geometry(n1, t1)[5], R_keep, pts)
        sol.remesh(n1, t1, ec, [], *boundary(n1, t1), nodal=(tets[ei], w), low=(np.c_[ec, ei[t1]], wake))
        nodes, tets = n1, t1
        xc, phi_n, r_nod, phi_c, r_c, uc = geometry(nodes, tets)
        slab = np.argmin(np.abs(r_c[:, None]-rs[None, :]), axis=1)
        surf = r_nod > Ro-surf_tol
        tree = cKDTree(nodes)
        n_base = len(tets)
        print(f"  -> rebuilt: {m0} -> {len(tets)} elements, {len(nodes)} nodes", flush=True)
        return sol.d.cpu().numpy()

    def adapt(sol):
        nonlocal nodes, tets, xc, phi_n, r_nod, phi_c, r_c, uc, slab, surf, tree
        pts = [tr["path"][-1] for tr in track.values()]
        dn = sol.d.cpu().numpy()
        if coarsen and pts and len(tets) > a.coarsen_grow*n_base:
            rebuild(sol, pts, dn)
            return

        def marked(r, dthr):
            hs = amr3d.size(nodes, tets)
            band = dn[tets].max(1) >= dthr
            if coarsen and pts:
                fine = near_tips(uc, r, pts) | (band & near_tips(uc, R_keep, pts))
                return (fine & (hs > h_ref)) | (band & (hs > hw_ref))
            return (near_tips(uc, r, pts) | band) & (hs > h_ref)

        if not marked(0.5*R_amr, a.amr_d).any():
            return
        m0, mids, parent = len(tets), [], np.arange(len(tets))
        while True:
            mark = marked(R_amr, 0.3*a.amr_d)
            if not mark.any():
                break
            nodes, tets, p, md = amr3d.refine(nodes, tets, mark)
            dn = amr3d.nodal(dn, md)
            xc, phi_n, r_nod, phi_c, r_c, uc = geometry(nodes, tets)
            parent = parent[p]
            mids += md
        sol.remesh(nodes, tets, parent, mids, *boundary(nodes, tets))
        slab = np.argmin(np.abs(r_c[:, None]-rs[None, :]), axis=1)
        surf = r_nod > Ro-surf_tol
        tree = cKDTree(nodes)
        print(f"  -> refined ({len(mids)} passes): {m0} -> {len(tets)} elements, {len(nodes)} nodes", flush=True)

    def hook(sol):
        d = sol.d.cpu().numpy()
        for root, tip, Q, far_ in find_cracks(d):
            key = next((k for k, tr in track.items() if np.hypot(*(Q-tr["path"][-1]).T).min() < 3*a.ell), None)
            if key is None:
                key = f"c{len(track)+1}"
                track[key] = {"root": root, "path": [root.copy()], "far": 0.0}
                print(f"  -> new crack {key} at (s={root[0]:.2f}, y={root[1]:.2f})", flush=True)
            tr = track[key]
            tr["far"] = far_
            if np.hypot(*(tip-tr["path"][-1])) > 0.5*a.h:
                tr["path"].append(tip.copy())
        th = {}
        if steer:
            U = torch.stack(sol.u_hist).cpu().numpy().reshape(len(sol.u_hist), -1, 3)
            if a.notch_steer:
                for side in (1, -1):
                    th["nR" if side > 0 else "nL"] = steer_notch(sol, side, U)
            for k, tr in track.items():
                if tr["far"] >= min_len:
                    th[k] = steer_tip(sol, k, U)
        far = {k: tr["far"] for k, tr in track.items()}
        edge = max((max(abs(tr["path"][-1][0]), 0.0) for tr in track.values()), default=0.0)
        edge_y = max((abs(tr["path"][-1][1]) for tr in track.values()), default=0.0)
        if (far and max(far.values()) >= a.s_stop) or edge > a.fine_s-2.0 or edge_y > a.fine_y-1.5:
            sol.ctl.n_cycles = 0
        json.dump({k: {"path": [list(map(float, p)) for p in v["path"]], "far": v["far"]} for k, v in track.items()},
                  open(out/"track.json", "w"))
        if a.amr:
            adapt(sol)
        return dict(n_cracks=len(track), s_max=max(far.values(), default=0.0),
                    **{f"s_{k}": v for k, v in far.items()}, **{f"th_{k}": v for k, v in th.items()})

    s.run(hooks=hook, restart=a.restart)


if __name__ == "__main__":
    main()
