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
from vctdpf.mesh import cruciform

DATA = ROOT/"data"/"zenodo_15807377"/"supplementaryMaterial"
EXP_PATH = {"I": DATA/"Experiment_I"/"job-013-ExpCrack001_00_crack-1_base.txt",
            "II": DATA/"Experiment_II"/"job-036-Wolf2019-LoadingTypeB_00_crack-1_base.txt",
            "A": DATA/"Experiment_A"/"job-041-FremyProportional"/"job-041-FremyProportional_00_crack-1_base.txt",
            "B": DATA/"Experiment_B"/"job-035-FremySquare"/"job-035-FremySquare_00_crack-1_base.txt",
            "C": DATA/"Experiment_C"/"job-040-FremyCross"/"job-040-FremyCross_00_crack-1_base.txt"}
for _e in ("D", "E", "T", "P", "Q"):
    EXP_PATH[_e] = EXP_PATH["A"]
STAGES = {"I": [(0.0, 0.0), (15.1, 22.5)],
          "II": [(0.0, 0.0), (7.34, -15.0), (9.32, -30.0), (12.14, -45.0), (14.76, -60.0), (17.77, -75.0),
                 (20.90, -90.0)],
          "A": [(0.0, 0.0)], "B": [(0.0, 0.0)], "C": [(0.0, 0.0)]}
FREMY = ("A", "B", "C", "D", "E", "T", "P", "Q")
A1 = {"I": 15.2, **{e: 18.0 for e in FREMY}}
STAGES.update({e: [(0.0, 0.0)] for e in FREMY})
_LO, _HI, _RT, _LT = (2650.0, 480.0), (3600.0, 5680.0), (5720.0, 3550.0), (500.0, 2620.0)
_C, _IP, _IM, _IIP, _IIM = (3100.0, 3070.0), (4600.0, 4580.0), (1600.0, 1550.0), (2050.0, 4150.0), (4100.0, 2000.0)
FREMY_PATH = {"A": [(0.0, *_LO), (0.5, *_HI), (1.0, *_LO)],
              "B": [(0.0, *_LO), (0.25, *_RT), (0.5, *_HI), (0.75, *_LT), (1.0, *_LO)],
              "C": [(0.0, *_C), (0.125, *_IP), (0.375, *_IM), (0.5, *_C), (0.625, *_IIP), (0.875, *_IIM), (1.0, *_C)]}
_ts = np.linspace(0.0, 1.0, 65)
_r = np.hypot(_LO[0]-_C[0], _LO[1]-_C[1])
FREMY_PATH["D"] = [(t, _C[0]+_r*np.cos(np.radians(-135)+2*np.pi*t), _C[1]+_r*np.sin(np.radians(-135)+2*np.pi*t))
                   for t in _ts]
FREMY_PATH["E"] = [(t, _C[0]+_r*np.cos(np.radians(-90)+2*np.pi*t), _C[1]+0.5*_r*np.sin(np.radians(-90)+2*np.pi*t))
                   for t in _ts]
FREMY_PATH["T"] = [(0.0, *_LO), (1/3, *_RT), (2/3, *_HI), (1.0, *_LO)]
FREMY_PATH["P"] = [(0.0, _C[0]-0.5*_r, _C[1]-0.9*_r), (0.5, _C[0]+0.5*_r, _C[1]+0.9*_r),
                   (1.0, _C[0]-0.5*_r, _C[1]-0.9*_r)]
_q = _r/np.sqrt(2)
FREMY_PATH["Q"] = [(0.0, _C[0]-_q, _C[1]-_q), (0.25, _C[0]+_q, _C[1]-_q), (0.5, _C[0]+_q, _C[1]+_q),
                   (0.75, _C[0]-_q, _C[1]+_q), (1.0, _C[0]-_q, _C[1]-_q)]
_CI, _CII = 1.505e-3, 1.93e-3
_HS_K = {"H01": [(8.4, 4.50), (14.0, 7.50)], "H03": [(14.0, 5.46), (14.0, 9.12)], "H04": [(14.0, -9.19), (14.0, 9.19)],
         "H05": [(8.4, 7.12), (14.0, 7.12)], "H06": [(14.0, 0.06), (14.0, 7.15)]}
for _e, _f in (("H01", 0.97/0.536), ("H03", 0.80/0.52), ("H05", 1.21/0.64), ("H06", 0.50/0.26)):
    _HS_K[_e + "t"] = [(k1, _f*k2) for k1, k2 in _HS_K[_e]]
for _e, ((_k1, _k2), (_K1, _K2)) in _HS_K.items():
    _v = [((s-d)/2, (s+d)/2) for s, d in ((_k1/_CI, _k2/_CII), (_K1/_CI, _K2/_CII))]
    FREMY_PATH[_e] = [(0.0, *_v[0]), (0.5, *_v[1]), (1.0, *_v[0])]
    EXP_PATH[_e] = EXP_PATH["A"]
FREMY = FREMY + tuple(_HS_K)
A1.update({e: 18.0 for e in _HS_K})
STAGES.update({e: [(0.0, 0.0)] for e in _HS_K})
HALF_THIN, R_THIN, T_IN, T_OUT = 71.48, 10.0, 1.0, 6.0
F_MAX, F_MIN = 52e3, 5.5e3


def thickness(xc: np.ndarray) -> np.ndarray:
    a = np.abs(xc)
    inside = (a[:, 0] <= HALF_THIN) & (a[:, 1] <= HALF_THIN)
    c = HALF_THIN-R_THIN
    corner = (a[:, 0] > c) & (a[:, 1] > c)
    inside &= ~corner | (np.hypot(a[:, 0]-c, a[:, 1]-c) <= R_THIN)
    return np.where(inside, T_IN, T_OUT)


def edge_forces(nodes: np.ndarray, sel: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    other = 1-axis
    idx = sel[np.argsort(nodes[sel, other])]
    s = nodes[idx, other]
    w = np.zeros(len(idx))
    seg = np.diff(s)
    w[:-1] += seg/2
    w[1:] += seg/2
    return idx, w/w.sum()


def initial_crack(exp: str, a_start: float | None, straight: bool = False) -> np.ndarray:
    P = np.loadtxt(EXP_PATH[exp])
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    if straight and a_start is not None:
        e = (P[1]-P[0])/np.linalg.norm(P[1]-P[0])
        tip = np.vstack([P[0], P[0]+a_start*e])
    elif exp == "II" and a_start is None:
        tip = P[:2]
    else:
        a1 = a_start if a_start is not None else A1[exp]
        k = int(np.searchsorted(s, a1+1e-6))
        tip = P[:k]
    return np.vstack([-tip[::-1], tip[1:]])


def nonproportionality(dI: np.ndarray, dII: np.ndarray, mode: str = "shape") -> float:
    x, y = np.r_[dI, dI[0]], np.r_[dII, dII[0]]
    area = 0.5*abs(np.sum(x[:-1]*y[1:]-x[1:]*y[:-1]))
    box = np.ptp(dI)**2 if mode == "mag" else np.ptp(dI)*np.ptp(dII)
    return float(min(1.0, 4.0*area/(np.pi*box))) if box > 0 else 0.0


def build(a):
    crack = initial_crack(a.exp, a.a_start, a.straight)
    if a.check_K or a.amr:
        x, y = crack[-1]
        r = a.amr_R*a.ell if a.amr else 3.0
        boxes = [(x-r, x+r, y-r, y+r), (-x-r, -x+r, -y-r, -y+r)]
    elif a.exp == "I" and a.screen:
        boxes = [(8.0, 24.0, 3.0, 20.0), (-24.0, -8.0, -20.0, -3.0)]
    elif a.exp == "I":
        boxes = [(8.0, 70.0, 2.0, 40.0), (-70.0, -8.0, -40.0, -2.0)]
    elif a.exp in FREMY:
        boxes = [(9.0, 22.0, 9.0, 22.0), (-22.0, -9.0, -22.0, -9.0)]
    else:
        boxes = [(0.0, 22.0, 0.0, 25.0), (-22.0, 0.0, -25.0, 0.0)]
    geom = "fremy_geom.npz" if a.exp in FREMY else "wolf_geom.npz"
    slit = trim(crack, a.precrack_band) if a.precrack_band > 0 else crack
    nodes, tris = cruciform(str(ROOT/"data"/geom), [slit], boxes, a.h, a.h_coarse,
                            slit_w=a.slit_w, h_mid=a.h_mid, mid_box=(-80, 80, -80, 80))
    return nodes, tris, crack


def trim(P, L):
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    q = np.r_[L, s[(s > L) & (s < s[-1]-L)], s[-1]-L]
    return np.c_[np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])]


def band_elements(nodes, tris, crack, L, half_w):
    xc = nodes[tris].mean(1)
    sel = np.zeros(len(tris), bool)
    for Q in (crack, crack[::-1]):
        S = trim(Q[::-1], 0.0)
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(S, axis=0).T))]
        a_, b_ = S[0], np.array([np.interp(L, s, S[:, 0]), np.interp(L, s, S[:, 1])])
        v = b_-a_
        u = np.clip(((xc-a_) @ v)/(v @ v), 0, 1)
        sel |= np.hypot(*(xc-a_-u[:, None]*v).T) < half_w
    return sel


def jitter(nodes, tris, frac, seed):
    e = np.sort(np.r_[tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    fixed = np.zeros(len(nodes), bool)
    fixed[ue[cnt == 1].ravel()] = True
    L = np.full(len(nodes), np.inf)
    le = np.hypot(*(nodes[ue[:, 0]] - nodes[ue[:, 1]]).T)
    np.minimum.at(L, ue[:, 0], le)
    np.minimum.at(L, ue[:, 1], le)
    rng = np.random.default_rng(seed)
    dx = rng.uniform(-1.0, 1.0, nodes.shape) * (frac * L)[:, None]
    dx[fixed] = 0.0
    return nodes + dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["I", "II", *FREMY], default="II")
    ap.add_argument("--phi", type=float, default=None,
                    help="Wolf: constant phase shift F_y behind F_x [deg] instead of the experimental stages")
    ap.add_argument("--ell", type=float, default=0.4)
    ap.add_argument("--h", type=float, default=0.15)
    ap.add_argument("--h-mid", type=float, default=2.0)
    ap.add_argument("--h-coarse", type=float, default=6.0)
    ap.add_argument("--slit-w", type=float, default=0.02)
    ap.add_argument("--a-start", type=float, default=None)
    ap.add_argument("--straight", action="store_true", help="straight initial crack of length a_start")
    ap.add_argument("--Gc", type=float, default=50.0)
    ap.add_argument("--plane", choices=["stress", "strain"], default="stress")
    ap.add_argument("--precrack-band", type=float, default=0.0,
                    help="> 0: the last L mm of the initial crack at each end is a band of elements with d -> 1 "
                         "imposed through the history field instead of a slit, as the pre-crack of tube3d.py")
    ap.add_argument("--alpha-T", type=float, default=None)
    ap.add_argument("--load-scale", type=float, default=1.0, help="scales F_m and F_a together")
    ap.add_argument("--substeps", type=int, default=16)
    ap.add_argument("--cycles", type=int, default=10**7)
    ap.add_argument("--dN-max", type=int, default=500)
    ap.add_argument("--dd-target", type=float, default=0.02)
    ap.add_argument("--df-target", type=float, default=0.02)
    ap.add_argument("--s-stop", type=float, default=None, help="stop at this crack arc length (mm)")
    ap.add_argument("--check-K", action="store_true", help="linear-elastic K_I at a_start by compliance")
    ap.add_argument("--fatigue", choices=["carrara", "chord"], default="carrara")
    ap.add_argument("--fat-deg", choices=["asym", "log"], default="asym", help="fatigue degradation function")
    ap.add_argument("--kappa", type=float, default=0.5, help="slope of the logarithmic degradation")
    ap.add_argument("--sigma-e", type=float, default=0.0, help="endurance limit [MPa] of the fatigue accumulation")
    ap.add_argument("--w-shear", type=float, default=1.0)
    ap.add_argument("--sy0", type=float, default=None, help="yield stress (enables J2 plasticity), 252 MPa")
    ap.add_argument("--H-iso", type=float, default=500.0)
    ap.add_argument("--H-kin", type=float, default=5000.0)
    ap.add_argument("--chi-p", type=float, default=1.0)
    ap.add_argument("--screen", action="store_true", help="Exp I: refine only near a1 (direction screening)")
    ap.add_argument("--restart", default=None, help="snapshot .npz in --out to continue from")
    ap.add_argument("--full-newton", action="store_true", help="refactorize at every plastic Newton iteration")
    ap.add_argument("--phase-sign", type=float, default=1.0, help="-1: phi_F -> -phi_F (mirror image problem)")
    ap.add_argument("--time-reverse", action="store_true", help="traverse every load cycle backwards in time")
    ap.add_argument("--t-shift", type=float, default=0.0, help="Fremy: start each cycle at this fraction of the path")
    ap.add_argument("--vctd-alpha", type=float, default=0.0,
                    help="anisotropy of the crack surface energy along the incremental-VCTD direction (0: off)")
    ap.add_argument("--vctd-R", type=float, default=5.0, help="radius of the steered tip zone, in units of ell")
    ap.add_argument("--vctd-w", type=float, default=3.0, help="half distance across the band for the jump, x ell")
    ap.add_argument("--vctd-star", action="store_true", help="VCTD* variant (Hengeveld Eq. 23-24)")
    ap.add_argument("--vctd-q", type=float, default=0.5, help="opening level for --vctd-open level")
    ap.add_argument("--vctd-open", choices=("mean", "contact", "level"), default="mean",
                    help="opening of the jump measured from the mid-value of the cycle, or the opening itself "
                         "(only the open crack contributes)")
    ap.add_argument("--vctd-odd", action="store_true",
                    help="steer by the time-reversal odd part of the VCTD direction only, with "
                         "A = I + alpha (e e - t t): A = I for every cycle that is its own time reversal")
    ap.add_argument("--vctd-dc", type=float, default=0.0,
                    help="with --vctd-odd: alpha = range(delta_I)/dc [mm], steering grows with the cyclic opening; "
                         "0: constant --vctd-alpha")
    ap.add_argument("--vctd-Kc", type=float, default=0.0,
                    help="with --vctd-odd: alpha = Delta K_I/Kc [MPa sqrt(mm)], Delta K_I from the opening range by "
                         "the plane-stress LEFM crack opening (8K/E) sqrt(r/2pi); independent of ell")
    ap.add_argument("--steer", choices=["vctd", "plastic"], default="vctd",
                    help="growth direction from the elastic VCTD or from the crack-tip plastic slip increments")
    ap.add_argument("--vctd-mu", type=float, default=float("inf"), help="Mode-II locking parameter mu (Eq. 20)")
    ap.add_argument("--vctd-frozen", action="store_true",
                    help="jump from an elastic sweep of the cycle at frozen damage (the crack geometry at the end "
                         "of the resolved cycle) instead of the displacements of the resolved cycle")
    ap.add_argument("--np-mode", choices=("shape", "mag"), default="shape",
                    help="non-proportionality measure for --np-weight, see nonproportionality()")
    ap.add_argument("--np-weight", action="store_true",
                    help="scale alpha by the non-proportionality of the cycle, phi = min(1, 4 A/(pi dI dII)) "
                         "(A: area enclosed by the jump path, dI, dII: ranges; |sin| of the phase lag for sines)")
    ap.add_argument("--vctd-rule", choices=("max", "sum"), default="max",
                    help="max: larger trial increment (Floros); sum: all increments of the cycle")
    ap.add_argument("--amr", action="store_true",
                    help="refine adaptively around the crack tips instead of a fixed refined box")
    ap.add_argument("--amr-R", type=float, default=8.0, help="radius refined to h around each tip, in units of ell")
    ap.add_argument("--amr-mode", choices=("track", "mirror"), default="track",
                    help="track: both tips tracked and steered independently, plus damage marking; "
                         "mirror: lower tip taken as the point reflection of the upper one (steered only above)")
    ap.add_argument("--amr-d", type=float, default=0.1,
                    help="track mode: elements with a nodal d above this are refined to h (Jaccon et al. 2023)")
    ap.add_argument("--steer-follow", action="store_true",
                    help="tip zones only (formed crack isotropic); redo a cycle whose crack leaves the steered zone")
    ap.add_argument("--follow-max", type=int, default=4, help="largest number of redos of one cycle")
    ap.add_argument("--steer-wake", choices=("iso", "keep", "band"), default="iso",
                    help="with --steer-follow, the tensor outside the tip zones: iso: reset to I every cycle; "
                         "keep: the last tensor stays; band: it stays only in the formed crack (element mean d >= 0.5)")
    ap.add_argument("--ridge-tangent", action="store_true", help="crack tangent from the damage ridge at the tip")
    ap.add_argument("--steer-smooth", type=float, default=1.0,
                    help="exponential average of the growth direction over about this many cycles (1: none)")
    ap.add_argument("--vctd-flip", action="store_true",
                    help="reflect the VCTD direction about the crack tangent (MTS side instead of slip side)")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="move interior nodes randomly by up to this fraction of their shortest edge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", action="store_true", help="print a cProfile summary of the run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.phi is not None:
        STAGES[a.exp] = [(0.0, a.phi)]
    out = Path(a.out or f"runs/wolf_{a.exp}_l{a.ell}_h{a.h}")
    out.mkdir(parents=True, exist_ok=True)

    snap_mesh = None
    if a.restart:
        mz = np.load(out/"mesh.npz")
        nodes, tris, crack = mz["nodes"], mz["tris"], mz["crack0"]
        zr = np.load(a.restart)
        if "nodes" in zr:
            snap_mesh = (zr["nodes"], zr["tris"], zr["thickness"])
            nodes, tris = snap_mesh[0], snap_mesh[1]
    else:
        nodes, tris, crack = build(a)
        if a.jitter > 0:
            nodes = jitter(nodes, tris, a.jitter, a.seed)
        if a.amr:
            tris = amr.label_longest(nodes, tris)
    xc = nodes[tris].mean(1)
    if snap_mesh is not None:
        th = snap_mesh[2]
    else:
        th = np.ones(len(tris)) if a.exp in FREMY else thickness(xc)
    print(f"mesh: {len(nodes)} nodes, {len(tris)} triangles, thin-zone elements {np.sum(th == T_IN)}", flush=True)
    Fm = 0.25*(F_MAX+F_MIN)*a.load_scale
    Fa = 0.25*(F_MAX-F_MIN)*a.load_scale
    st = {"phi": 22.5 if a.exp == "I" else 0.0, "stage": 0}
    sg = -1.0 if a.time_reverse else 1.0
    if a.exp in FREMY:
        V = np.array(FREMY_PATH[a.exp])
        cx, cy = (1, 2) if a.phase_sign > 0 else (2, 1)
        ampx = lambda t: a.load_scale*np.interp((sg*np.asarray(t)+a.t_shift) % 1.0, V[:, 0], V[:, cx])
        ampy = lambda t: a.load_scale*np.interp((sg*np.asarray(t)+a.t_shift) % 1.0, V[:, 0], V[:, cy])
    else:
        ampx = lambda t: Fm+Fa*np.sin(sg*2*np.pi*t)
        ampy = lambda t: Fm+Fa*np.sin(sg*2*np.pi*t-np.radians(a.phase_sign*st["phi"]))

    def boundary(nodes):
        tol = 1e-6
        xmin, xmax, ymin, ymax = nodes[:, 0].min(), nodes[:, 0].max(), nodes[:, 1].min(), nodes[:, 1].max()
        left = np.flatnonzero(nodes[:, 0] < xmin+tol)
        bot = np.flatnonzero(nodes[:, 1] < ymin+tol)
        right = np.flatnonzero(nodes[:, 0] > xmax-tol)
        top = np.flatnonzero(nodes[:, 1] > ymax-tol)
        ir, wr = edge_forces(nodes, right, 0)
        it, wt = edge_forces(nodes, top, 1)
        ch = [Channel("force", 2*ir, wr, ampx, "Fx"), Channel("force", 2*it+1, wt, ampy, "Fy")]
        return ch, np.r_[2*left, 2*bot+1]

    ch, fixed = boundary(nodes)

    mat = Material(E=192e3, nu=0.24, Gc=a.Gc, ell=a.ell, plane=a.plane, alpha_T=a.alpha_T,
                   fatigue=a.fatigue, fat_deg=a.fat_deg, kappa=a.kappa, sigma_e=a.sigma_e, w_shear=a.w_shear, sy0=a.sy0,
                   H_iso=a.H_iso, H_kin=a.H_kin, chi_p=a.chi_p)
    s_stop = a.s_stop or {"I": 60.0, "II": 22.5}.get(a.exp, 21.0)
    ctl = Controls(substeps=a.substeps, n_cycles=a.cycles, dN_max=a.dN_max, dd_target=a.dd_target,
                   df_target=a.df_target, out=str(out), save_every=10, reuse_factor=not a.full_newton)
    if a.check_K:
        check_K(nodes, tris, th, mat, ch, fixed, ctl, crack, a)
        return
    s = FatigueSolver(nodes, tris, mat, ch, fixed, ctl, thickness=th)
    if snap_mesh is not None:
        s.n_remesh = 1
    elif a.precrack_band > 0:
        import torch
        pc = band_elements(nodes, tris, crack, a.precrack_band, 0.75*a.h)
        s.H[torch.as_tensor(np.flatnonzero(pc), device=s.H.device)] = 500.0*a.Gc/a.ell
        print(f"pre-crack band: {int(pc.sum())} elements over {a.precrack_band:g} mm at each end", flush=True)
    print(f"setup {s.t_setup:.1f}s, dofs {2*len(nodes)}, Fm={Fm:.0f} N, Fa={Fa:.0f} N, "
          f"alpha_T={mat.aT():.3g}", flush=True)
    tip0 = crack[-1].copy()
    track = {"tips": [tip0], "lo": [crack[0].copy()],
             "s": float(np.sum(np.hypot(*np.diff(crack[len(crack)//2:], axis=0).T)))}
    both = a.amr_mode == "track"
    if a.restart:
        tj = json.load(open(out/"track.json"))
        track = {"tips": [np.array(p) for p in tj["tips"]], "s": tj["s"],
                 "lo": [np.array(p) for p in tj.get("lo", [crack[0]])]}
        st.update(tj["st"])
    else:
        k0 = max(i for i, (s0, _) in enumerate(STAGES[a.exp]) if track["s"] >= s0-1e-9)
        st.update(stage=k0, phi=STAGES[a.exp][k0][1])
        print(f"initial crack arc length {track['s']:.2f} mm, stage {k0}, phi_F = {st['phi']:.1f} deg", flush=True)
        np.savez_compressed(out/"mesh.npz", nodes=nodes, tris=tris, thickness=th, crack0=crack)
        json.dump(vars(a), open(out/"args.json", "w"), indent=1)

    steer = a.vctd_alpha > 0 or a.vctd_dc > 0 or a.vctd_Kc > 0
    if steer:
        from scipy.spatial import cKDTree
        from vctdpf.criteria import vctd, vctd_odd
        import torch
        s.record_u = True
        s.keep_checkpoint = a.steer_follow
        tree = cKDTree(nodes)
        if s.A_el is None:
            s.A_el = torch.eye(2, dtype=torch.float64, device=s.d.device).repeat(len(tris), 1, 1)
        tdir0 = (crack[-1]-crack[-2])/np.linalg.norm(crack[-1]-crack[-2])
        vlog = open(out/"vctd.txt", "a")
        vlog_lo = open(out/"vctd_lo.txt", "a") if both else None
        plog = open(out/"plastic_ctd.txt", "a")
        if a.steer == "plastic" and a.sy0 is None:
            raise SystemExit("--steer plastic needs --sy0")

    steer_state = {}

    def crack_tangent(key="tips", sol=None):
        tips = track[key]
        p = tips[-1]
        if sol is not None and a.ridge_tangent:
            d = sol.d.cpu().numpy()
            r = np.hypot(*(nodes-p).T)
            k = (d >= 0.9) & (r < 2.5*a.ell)
            if k.sum() >= 6:
                X, w = nodes[k], d[k]
                c = (X*w[:, None]).sum(0)/w.sum()
                Y = X-c
                v = np.linalg.eigh((Y*w[:, None]).T@Y)[1][:, -1]
                if v@(p-c) < 0:
                    v = -v
                return v
        back = [q for q in tips if np.hypot(*(p-q)) >= 1.0]
        if not back:
            return tdir0 if key == "tips" else -tdir0
        v = p-back[-1]
        return v/np.linalg.norm(v)

    def plastic_slip_direction(sol, tip, tdir, ndir, dI_total):
        rel = xc-tip
        xi, eta = rel@tdir, rel@ndir
        L, W = 3.0*a.ell, a.vctd_w*a.ell
        de = sol.d_elem().cpu().numpy()
        strip = np.flatnonzero((xi > -a.ell) & (xi < 2.0*a.ell) & (np.abs(eta) < W) & (de < 0.5))
        idx = torch.as_tensor(strip, device=sol.d.device)
        dep = torch.stack(sol.dep_hist)[:, idx].cpu().numpy()
        A = sol.area[idx].cpu().numpy()
        n0, n1, t0, t1 = ndir[0], ndir[1], tdir[0], tdir[1]
        enn = dep[..., 0]*n0**2+dep[..., 1]*n1**2+dep[..., 2]*n0*n1
        gtn = 2*(dep[..., 0]*t0*n0+dep[..., 1]*t1*n1)+dep[..., 2]*(t0*n1+t1*n0)
        dIp, dIIp = (enn*A).sum(1)/L, (gtn*A).sum(1)/L
        ap_ = np.clip(dIp, 0, None)
        m = np.sqrt(ap_**2+2*ap_*np.abs(dIIp)+2*dIIp**2)
        th_k = np.arcsin(np.clip(np.divide(dIIp, m, out=np.zeros_like(m), where=m > 0), -1, 1))
        wk = m*((dI_total > 0) & (dIp > 0))
        v = np.array([(wk*np.cos(th_k)).sum(), (wk*np.sin(th_k)).sum()])
        plog.write(" ".join(f"{x:.4e}" for x in (*dIp, *dIIp))+"\n")
        plog.flush()
        return float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 0 else None

    def apply_zone(sol, key):
        z = steer_state["Z_"+key]
        rel = xc-z["tip"]
        near = np.logical_or.reduce([np.hypot(*(xc-p).T) < a.vctd_R*a.ell for p in z["pts"]])
        idx = torch.as_tensor(np.flatnonzero(near & (rel@z["tdir"] > -a.ell)), device=sol.d.device)
        sol.A_el[idx] = z["A"]

    def front_tip(sol, key):
        d = sol.d.cpu().numpy()
        side = nodes[:, 0]+nodes[:, 1]
        on = (d > 0.9) & ((side > 0) if key == "tips" else (side < 0))
        if not on.any():
            return None
        P = nodes[on]
        return P[np.argmax(np.hypot(P[:, 0], P[:, 1]))]

    def overrun(sol):
        if steer_state.get("n_redo", 0) >= a.follow_max:
            return False
        moved = []
        for key in ("tips", "lo") if both else ("tips",):
            z = steer_state.get("Z_"+key)
            q = front_tip(sol, key)
            if z is None or q is None:
                continue
            if min(np.hypot(*(q-p)) for p in z["pts"]) > (a.vctd_R-1.0)*a.ell:
                moved.append((key, q))
        if not moved:
            return False
        sol.restore(sol._ck)
        steer_state["n_redo"] = steer_state.get("n_redo", 0)+1
        for key, q in moved:
            z = steer_state["Z_"+key]
            p0 = z["pts"][-1]
            n = int(np.ceil(np.hypot(*(q-p0))/a.ell))+1
            z["pts"] += [p0+(q-p0)*k/n for k in range(1, n+1)]
        if a.amr:
            adapt(sol, track["tips"][-1], extra=[q for _, q in moved])
        for key in ("tips", "lo") if both else ("tips",):
            if "Z_"+key in steer_state:
                apply_zone(sol, key)
        print(f"  -> crack left the steered zone; cycle redone with the zone along the advance "
              f"({steer_state['n_redo']})", flush=True)
        sol.redo = True
        return True

    def steer_tip(sol, tip, key="tips", U=None):
        upper = key == "tips"
        tdir = crack_tangent(key, sol)
        ndir = np.array([-tdir[1], tdir[0]])
        if U is None:
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
        if upper and not steer_state.get("dumped"):
            np.savetxt(out/"jump_cycle.txt", np.c_[dI, dII], header="delta_I delta_II (first steering call)")
            steer_state["dumped"] = True
        if a.steer == "plastic":
            th = plastic_slip_direction(sol, tip, tdir, ndir, dI)
        else:
            th = (vctd_odd if a.vctd_odd else vctd)(dI, dII, star=a.vctd_star, mu=a.vctd_mu, rule=a.vctd_rule,
                                                     opening=a.vctd_open, q=a.vctd_q)[0]
            if th is not None and a.vctd_flip:
                th = -th
        e_prev = steer_state.get("e" if upper else "e_lo")
        if th is None:
            e = tdir if e_prev is None else e_prev
        else:
            e = np.cos(th)*tdir+np.sin(th)*ndir
            if e_prev is not None and a.steer_smooth > 1:
                e = e_prev+(e-e_prev)/a.steer_smooth
                e /= np.linalg.norm(e)
        th = math.atan2(e@ndir, e@tdir)
        steer_state["e" if upper else "e_lo"] = e
        et = torch.as_tensor(e, dtype=torch.float64, device=sol.d.device)
        phi = nonproportionality(dI, dII, a.np_mode) if a.np_weight else 1.0
        if a.vctd_odd:
            tt = torch.as_tensor(tdir, dtype=torch.float64, device=sol.d.device)
            dc = a.vctd_Kc*8.0*np.mean(np.sqrt(rbs/(2*np.pi)))/192e3 if a.vctd_Kc > 0 else a.vctd_dc
            al = np.ptp(dI)/dc if dc > 0 else a.vctd_alpha
            Sd = al*(torch.outer(et, et)-torch.outer(tt, tt))
            lam = al*abs(math.sin(th))
            sh = math.sinh(lam)/lam if lam > 1e-12 else 1.0
            Az = math.cosh(lam)*torch.eye(2, dtype=torch.float64, device=sol.d.device)+sh*Sd
        else:
            Az = torch.eye(2, dtype=torch.float64, device=sol.d.device)+a.vctd_alpha*phi*torch.outer(et, et)
        steer_state["Z_"+key] = dict(A=Az, tip=tip.copy(), tdir=tdir, pts=[tip.copy()])
        apply_zone(sol, key)
        (vlog if upper else vlog_lo).write(f"{tip[0]:.4f} {tip[1]:.4f} {math.degrees(math.atan2(tdir[1], tdir[0])):.2f} "
                   f"{math.degrees(th):.2f} {dI.min():.4e} {dI.max():.4e} {dII.min():.4e} {dII.max():.4e} "
                   f"{phi:.4f}\n")
        (vlog if upper else vlog_lo).flush()
        return math.degrees(th)

    def adapt(sol, tip, extra=()):
        nonlocal nodes, tris, xc, tree
        R = a.amr_R*a.ell
        tips = [tip, track["lo"][-1] if both else -tip, *extra]
        dn = sol.d.cpu().numpy()

        def near(r):
            return np.logical_or.reduce([np.hypot(*(xc-p).T) < r for p in tips])

        def damaged(t, dn):
            return dn[t].max(1) >= a.amr_d if both else np.zeros(len(t), bool)

        if not ((near(0.5*R) | damaged(tris, dn)) & (amr.size(nodes, tris) > a.h)).any():
            return 0
        m0, mids = len(tris), []
        parent = np.arange(m0)
        while True:
            mark = (near(R) | damaged(tris, dn)) & (amr.size(nodes, tris) > a.h)
            if not mark.any():
                break
            nodes, tris, p, ea, eb = amr.refine(nodes, tris, mark)
            dn = np.r_[dn, 0.5*(dn[ea]+dn[eb])]
            xc = nodes[tris].mean(1)
            parent = parent[p]
            mids.append((ea, eb))
        ch_, fixed_ = boundary(nodes)
        sol.remesh(nodes, tris, parent, mids, ch_, fixed_)
        if steer:
            tree = cKDTree(nodes)
        print(f"  -> refined ({len(mids)} passes): {m0} -> {len(tris)} elements", flush=True)
        return len(mids)

    def hook(sol):
        if steer and a.steer_follow and sol._ck is not None and overrun(sol):
            return {}
        steer_state["n_redo"] = 0
        d = sol.d.cpu().numpy()
        on = (d > 0.9) & (nodes[:, 0]+nodes[:, 1] > 0)
        if on.any():
            P = nodes[on]
            r = np.hypot(P[:, 0], P[:, 1])
            tip = P[np.argmax(r)]
            prev = track["tips"][-1]
            step = float(np.hypot(*(tip-prev)))
            if np.hypot(*tip) > np.hypot(*prev)+0.5*a.h:
                track["tips"].append(tip)
                track["s"] += step
        lo = (d > 0.9) & (nodes[:, 0]+nodes[:, 1] < 0)
        if both and lo.any():
            P = nodes[lo]
            q = P[np.argmax(np.hypot(P[:, 0], P[:, 1]))]
            if np.hypot(*q) > np.hypot(*track["lo"][-1])+0.5*a.h:
                track["lo"].append(q)
        tip = track["tips"][-1]
        U = None
        if steer and a.vctd_frozen:
            U = sol.elastic_cycle(a.substeps).cpu().numpy().reshape(a.substeps, -1, 2)
        if steer and a.steer_follow and a.steer_wake != "keep":
            I2 = torch.eye(2, dtype=torch.float64, device=sol.d.device)
            if a.steer_wake == "iso":
                sol.A_el[:] = I2
            else:
                de = sol.d[torch.as_tensor(tris, device=sol.d.device)].mean(1)
                sol.A_el[de < 0.5] = I2
        th_c = steer_tip(sol, tip, U=U) if steer else 0.0
        if steer and both:
            steer_tip(sol, track["lo"][-1], "lo", U=U)
        if a.amr:
            adapt(sol, tip)
        stages = STAGES[a.exp]
        k = max(i for i, (s0, _) in enumerate(stages) if track["s"] >= s0-1e-9)
        if k != st["stage"]:
            st["stage"], st["phi"] = k, stages[k][1]
            print(f"  -> stage {k}: phi_F = {st['phi']:.0f} deg at s = {track['s']:.2f} mm", flush=True)
        if track["s"] >= s_stop:
            sol.ctl.n_cycles = 0
        np.savetxt(out/"tip_path.txt", np.array(track["tips"]))
        np.savetxt(out/"tip_path_lo.txt", np.array(track["lo"]))
        json.dump({"tips": [list(map(float, p)) for p in track["tips"]], "s": track["s"], "st": st,
                   "lo": [list(map(float, p)) for p in track["lo"]]},
                  open(out/"track.json", "w"))
        return dict(tip_x=float(tip[0]), tip_y=float(tip[1]), s=track["s"], phi=st["phi"], theta_c=th_c)

    if a.profile:
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.enable()
        s.run(hooks=hook, restart=a.restart)
        pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(40)
    else:
        s.run(hooks=hook, restart=a.restart)


def check_K(nodes, tris, th, mat, ch, fixed, ctl, crack, a):
    mat.split = "none"
    tip = crack[-1]
    tdir = (crack[-1]-crack[-2])/np.linalg.norm(crack[-1]-crack[-2])
    ndir = np.array([-tdir[1], tdir[0]])
    rel = nodes-tip
    xi, eta = rel@tdir, rel@ndir
    face = (xi < -0.05) & (xi > -1.5) & (np.abs(eta) < a.slit_w)
    up, lo = np.flatnonzero(face & (eta > 0)), np.flatnonzero(face & (eta < 0))
    s = FatigueSolver(nodes, tris, mat, ch, fixed, ctl, thickness=th)
    print(f"tip {tip.round(3)}, face nodes {len(up)}/{len(lo)}", flush=True)
    rows = []
    for t in np.linspace(0, 1, 17)[:-1]:
        s.solve_u(float(t))
        u = s.u.cpu().numpy().reshape(-1, 2)
        Ks = []
        for comp in (ndir, tdir):
            ru, du = -xi[up], u[up]@comp
            rl, dl = -xi[lo], u[lo]@comp
            r = np.linspace(0.1, 1.2, 12)
            jump = np.interp(r, np.sort(ru), du[np.argsort(ru)])-np.interp(r, np.sort(rl), dl[np.argsort(rl)])
            Kr = mat.E/8*np.sqrt(2*np.pi/r)*jump
            Ks.append(np.polyval(np.polyfit(r, Kr, 1), 0.0)/math.sqrt(1000))
        rows.append((t, *Ks))
        print(f"  t={t:.4f}  K_I={Ks[0]:7.2f}  K_II={Ks[1]:7.2f} MPa sqrt(m)", flush=True)
    np.savetxt(Path(ctl.out)/"K_cycle.txt", np.array(rows), header="t KI KII [MPa sqrt(m)]")


if __name__ == "__main__":
    main()
