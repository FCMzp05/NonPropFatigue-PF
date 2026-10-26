from __future__ import annotations

import csv
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import nvmath.sparse.advanced as nsa

from .plasticity import J2PlaneStress

try:
    from .j2_kernel import return_map_element as _j2_elem
except ImportError:
    _j2_elem = None

warnings.filterwarnings("ignore", message=".*Sparse.*")
DEV = torch.device("cpu" if os.environ.get("PFF_DEVICE") == "cpu" else "cuda")
F64 = torch.float64


class _Pardiso:
    def __init__(self, upper: torch.Tensor, b: torch.Tensor):
        import pypardiso
        self.vals, self.b, self.n = upper.values(), b, upper.shape[0]
        self.crow = upper.crow_indices().numpy().astype(np.int32)
        self.col = upper.col_indices().numpy().astype(np.int32)
        self.ps = pypardiso.PyPardisoSolver(mtype=2)

    def plan(self):
        pass

    def factorize(self):
        from scipy.sparse import csr_matrix
        self.M = csr_matrix((self.vals.numpy().copy(), self.col, self.crow), shape=(self.n, self.n))
        self.ps.factorize(self.M)

    def solve(self) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.ps.solve(self.M, self.b.numpy().ravel()), dtype=np.float64)).view(-1, 1)

    def free(self):
        self.ps.free_memory(everything=True)


def _direct_solver(upper, b, opt):
    return _Pardiso(upper, b) if DEV.type == "cpu" else nsa.DirectSolver(upper, b, options=opt, execution="cuda")


def T_(a, dtype=F64):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=DEV)


class Pattern:
    shift32 = 1e-6

    def __init__(self, tris: np.ndarray, n_nodes: int, nd: int):
        dofs = (tris[:, :, None]*nd+np.arange(nd)).reshape(len(tris), -1)
        k = dofs.shape[1]
        rows = np.repeat(dofs, k, axis=1).ravel()
        cols = np.tile(dofs, (1, k)).ravel()
        n = n_nodes*nd
        key = rows.astype(np.int64)*n+cols
        uniq, inv = np.unique(key, return_inverse=True)
        r, c = uniq//n, uniq % n
        indptr = np.zeros(n+1, dtype=np.int64)
        np.add.at(indptr, r+1, 1)
        indptr = np.cumsum(indptr)
        self.n, self.k, self.nnz = n, k, len(uniq)
        self.inv = T_(inv, torch.int64)
        self.rows, self.cols = T_(r, torch.int64), T_(c, torch.int64)
        self.full = torch.sparse_csr_tensor(T_(indptr, torch.int64), T_(c, torch.int64),
                                            torch.zeros(len(uniq), dtype=F64, device=DEV), size=(n, n))
        self.data = self.full.values()
        up = np.flatnonzero(c >= r)
        ru, cu = r[up], c[up]
        iu = np.zeros(n+1, dtype=np.int32)
        np.add.at(iu, ru+1, 1)
        iu = np.cumsum(iu).astype(np.int32)
        self.up = T_(up, torch.int64)
        self.upper = torch.sparse_csr_tensor(T_(iu, torch.int32), T_(cu, torch.int32),
                                             torch.zeros(len(up), dtype=F64, device=DEV), size=(n, n))
        self.udata = self.upper.values()
        self.ru, self.cu = T_(ru, torch.int64), T_(cu, torch.int64)
        self.diag_u = T_(np.flatnonzero(ru == cu), torch.int64)
        self.udata[self.diag_u] = 1.0
        self.b = torch.zeros((n, 1), dtype=F64, device=DEV)
        opt = nsa.DirectSolverOptions(sparse_system_type=nsa.DirectSolverMatrixType.SPD,
                                      sparse_system_view=nsa.DirectSolverMatrixViewType.UPPER)
        self.solver = _direct_solver(self.upper, self.b, opt)
        self.solver.plan()

    def free(self) -> None:
        self.solver.free()
        if hasattr(self, "solver32"):
            self.solver32.free()

    def assemble(self, ke: torch.Tensor) -> None:
        self.data.zero_()
        self.data.index_add_(0, self.inv, ke.reshape(-1))

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return (self.full@x.view(-1, 1)).view(-1)

    def _upper(self, fixed: torch.Tensor) -> None:
        if getattr(self, "_fixed", None) is not fixed:
            free = ~fixed
            self._keep = (free[self.ru] & free[self.cu]).to(F64)
            self._fix_diag = fixed[self.ru[self.diag_u]]
            self._fixed = fixed
        self.udata.copy_(self.data[self.up])
        self.udata.mul_(self._keep)
        self.udata[self.diag_u] = torch.where(self._fix_diag, 1.0, self.udata[self.diag_u])

    def factor(self, fixed: torch.Tensor) -> None:
        self._upper(fixed)
        self.solver.factorize()

    def factor32(self, fixed: torch.Tensor) -> None:
        if not hasattr(self, "solver32"):
            self.upper32 = torch.sparse_csr_tensor(self.upper.crow_indices(), self.upper.col_indices(),
                                                   torch.zeros(len(self.udata), dtype=torch.float32, device=DEV),
                                                   size=self.upper.shape)
            self.udata32 = self.upper32.values()
            self.b32 = torch.zeros((self.n, 1), dtype=torch.float32, device=DEV)
            opt = nsa.DirectSolverOptions(sparse_system_type=nsa.DirectSolverMatrixType.SPD,
                                          sparse_system_view=nsa.DirectSolverMatrixViewType.UPPER)
            self.solver32 = _direct_solver(self.upper32, self.b32, opt)
            self.solver32.plan()
        self._upper(fixed)
        self._s = self.udata[self.diag_u].rsqrt()
        self.udata32.copy_(self.udata*self._s[self.ru]*self._s[self.cu])
        self.udata32[self.diag_u] += self.shift32
        self.solver32.factorize()

    def back32(self, rhs: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        self.b32.copy_((torch.where(fixed, 0.0, rhs)*self._s).view(-1, 1))
        return self.solver32.solve().view(-1).to(F64)*self._s

    def pcg(self, rhs: torch.Tensor, fixed: torch.Tensor, x0: torch.Tensor, tol: float, maxit: int, prec=None):
        back = prec or self.back
        free = (~fixed).to(F64)
        A = lambda x: free*self.matvec(free*x)+(1.0-free)*x
        b = free*rhs
        x = free*x0
        r = b-A(x)
        bn = float(torch.linalg.norm(b))
        z = back(r, fixed)
        p, rz = z, torch.dot(r, z)
        for k in range(maxit):
            if float(torch.linalg.norm(r)) <= tol*bn:
                return x, k
            Ap = A(p)
            al = rz/torch.dot(p, Ap)
            x = x+al*p
            r = r-al*Ap
            z = back(r, fixed)
            rz_new = torch.dot(r, z)
            p, rz = z+(rz_new/rz)*p, rz_new
        return None, maxit

    def back(self, rhs: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        self.b.copy_(torch.where(fixed, 0.0, rhs).view(-1, 1))
        return self.solver.solve().view(-1).clone()

    def solve(self, rhs: torch.Tensor, fixed: torch.Tensor | None = None,
              u_fixed: torch.Tensor | None = None) -> torch.Tensor:
        r = rhs.clone()
        self.udata.copy_(self.data[self.up])
        if fixed is not None:
            if u_fixed is not None and bool(torch.any(u_fixed != 0)):
                xc = torch.zeros(self.n, dtype=F64, device=DEV)
                xc[fixed] = u_fixed[fixed]
                r -= self.matvec(xc)
                r[fixed] = u_fixed[fixed]
            else:
                r[fixed] = 0.0
            free = ~fixed
            self.udata.mul_(free[self.ru] & free[self.cu])
            dg = self.ru[self.diag_u]
            self.udata[self.diag_u] = torch.where(fixed[dg], torch.ones_like(self.udata[self.diag_u]),
                                                  self.udata[self.diag_u])
        self.solver.factorize()
        self.b.copy_(r.view(-1, 1))
        return self.solver.solve().view(-1).clone()


@dataclass
class Material:
    E: float
    nu: float
    Gc: float
    ell: float
    plane: str = "stress"
    k_res: float = 1e-6
    alpha_T: float | None = None
    split: str = "amor"
    fatigue: str = "carrara"
    fat_deg: str = "asym"
    kappa: float = 0.5
    sigma_e: float = 0.0
    w_shear: float = 1.0
    sy0: float | None = None
    H_iso: float = 0.0
    H_kin: float = 0.0
    chi_p: float = 1.0

    def lame2(self):
        mu = self.E/(2*(1+self.nu))
        lam = self.E*self.nu/(1-self.nu**2) if self.plane == "stress" else \
            self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        return lam, mu

    def aT(self):
        return self.alpha_T if self.alpha_T is not None else self.Gc/(12*self.ell)


@dataclass
class Channel:
    kind: str
    dofs: np.ndarray
    ref: np.ndarray
    amplitude: Callable[[np.ndarray], np.ndarray]
    name: str = ""


@dataclass
class Controls:
    substeps: int = 24
    max_stagger: int = 30
    stagger_tol: float = 1e-4
    pcg_tol: float = 0.0
    pcg_maxit: int = 60
    omega: float = 1.0
    max_mask_iter: int = 20
    n_cycles: int = 10**6
    dN_init: int = 1
    dN_max: int = 1000
    dd_target: float = 0.02
    df_target: float = 0.02
    d_stop: float = 0.95
    save_every: int = 10
    lag_mask: bool = True
    mask_cap: int = 0
    reuse_factor: bool = True
    reuse_rate: float = 0.1
    out: str = "out"


class FatigueSolver:
    def __init__(self, nodes: np.ndarray, tris: np.ndarray, mat: Material, channels: list[Channel],
                 fixed_dofs: np.ndarray, ctl: Controls, thickness: np.ndarray | float = 1.0,
                 d0: np.ndarray | None = None, stop_nodes: np.ndarray | None = None):
        self.mat, self.ctl, self.channels = mat, ctl, channels
        self.nodes, self.tris = nodes, tris
        self.thickness = np.broadcast_to(np.asarray(thickness, dtype=float), (len(tris),)).copy()
        self.n_remesh, self._remap = getattr(self, "n_remesh", 0), None
        self.keep_checkpoint, self.redo, self._ck = getattr(self, "keep_checkpoint", False), False, None
        n, m = len(nodes), len(tris)
        self.n, self.m = n, m
        x = nodes[tris]
        J = np.stack([x[:, 1]-x[:, 0], x[:, 2]-x[:, 0]], axis=2)
        det = J[:, 0, 0]*J[:, 1, 1]-J[:, 0, 1]*J[:, 1, 0]
        if np.any(det <= 0):
            raise ValueError("non-positive element orientation")
        Jinv = np.linalg.inv(J)
        dN = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        G = dN@Jinv
        th = np.broadcast_to(np.asarray(thickness, dtype=float), (m,))
        self.vol = T_(0.5*det*th)
        self.area = T_(0.5*det)
        self.G = T_(G)
        B = np.zeros((m, 3, 6))
        B[:, 0, 0::2] = G[:, :, 0]
        B[:, 1, 1::2] = G[:, :, 1]
        B[:, 2, 0::2] = G[:, :, 1]
        B[:, 2, 1::2] = G[:, :, 0]
        self.B = T_(B)
        self.tri_t = T_(tris, torch.int64)
        self.dof_t = (self.tri_t[:, :, None]*2+torch.arange(2, device=DEV)).reshape(m, 6)
        t0 = time.time()
        self.Pu = Pattern(tris, n, 2)
        self.Pd = Pattern(tris, n, 1)
        self.t_setup = time.time()-t0
        lam, mu = mat.lame2()
        self.K2, self.mu = lam+mu, mu
        I = torch.tensor([[1.0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=F64, device=DEV)
        Pd = mu*torch.tensor([[1.0, -1, 0], [-1, 1, 0], [0, 0, 1]], dtype=F64, device=DEV)
        self.CI, self.CD = self.K2*I, Pd
        self.fixed = torch.zeros(2*n, dtype=torch.bool, device=DEV)
        self.fixed[T_(fixed_dofs, torch.int64)] = True
        self.ch = []
        for c in channels:
            d = T_(c.dofs, torch.int64)
            ref = T_(c.ref)
            if c.kind == "disp":
                self.fixed[d] = True
            self.ch.append((c, d, ref))
        self.u = torch.zeros(2*n, dtype=F64, device=DEV)
        self.d = torch.zeros(n, dtype=F64, device=DEV) if d0 is None else T_(d0)
        self.H = torch.zeros(m, dtype=F64, device=DEV)
        self.abar = torch.zeros(m, dtype=F64, device=DEV)
        self.alpha_prev = torch.zeros(m, dtype=F64, device=DEV)
        self.comp = torch.zeros(m, dtype=torch.bool, device=DEV)
        self.eps_hist: list[torch.Tensor] = []
        self.pl = None
        self.debug = False
        self.u_pred, self.u_prev = None, None
        self._fact_ok, self.n_fact, self.n_newton = False, 0, 0
        self.A_el: torch.Tensor | None = None
        self.record_u, self.u_hist = False, []
        self.dep_hist: list[torch.Tensor] = []
        self.KD_e = self.KI_e = None
        self.mask_changed = False
        self.has_disp = any(c.kind == "disp" for c in channels)
        if mat.sy0 is not None:
            if mat.plane != "stress":
                raise ValueError("plasticity is implemented for plane stress only")
            self.pl = J2PlaneStress(mat.E, mat.nu, mat.sy0, mat.H_iso, mat.H_kin).to(DEV)
            z3 = torch.zeros(m, 3, dtype=F64, device=DEV)
            self.ep, self.beta, self.alp = z3.clone(), z3.clone(), torch.zeros(m, dtype=F64, device=DEV)
            self.ep_t, self.beta_t, self.alp_t, self.sig_t = z3.clone(), z3.clone(), self.alp.clone(), z3.clone()
            self.Wp = torch.zeros(m, dtype=F64, device=DEV)
        self.stop_nodes = None if stop_nodes is None else T_(stop_nodes, torch.int64)
        Me = torch.full((3, 3), 1.0/12.0, dtype=F64, device=DEV)+torch.eye(3, dtype=F64, device=DEV)/12.0
        self.Me = Me
        self.GG = torch.einsum("eia,eja->eij", self.G, self.G)

    def remesh(self, nodes: np.ndarray, tris: np.ndarray, parent: np.ndarray, mids: list[tuple],
               channels: list[Channel], fixed_dofs: np.ndarray):
        pa = T_(parent, torch.int64)
        passes = [(T_(ea, torch.int64), T_(eb, torch.int64)) for ea, eb in mids]

        def nod(x):
            for ia, ib in passes:
                x = torch.cat([x, 0.5*(x[ia]+x[ib])])
            return x

        keep = dict(d=nod(self.d), u=nod(self.u.view(-1, 2)).reshape(-1),
                    u_prev=None if self.u_prev is None else nod(self.u_prev.view(-1, 2)).reshape(-1),
                    H=self.H[pa], abar=self.abar[pa], alpha_prev=self.alpha_prev[pa], comp=self.comp[pa],
                    A_el=None if self.A_el is None else self.A_el[pa])
        if self.pl is not None:
            for k in ("ep", "beta", "alp", "Wp", "sig_t"):
                keep[k] = getattr(self, k)[pa]
        flags = dict(record_u=self.record_u, debug=self.debug, n_fact=self.n_fact, n_newton=self.n_newton,
                     n_remesh=self.n_remesh+1)
        stop = None if self.stop_nodes is None else self.stop_nodes.cpu().numpy()
        self.Pu.free()
        self.Pd.free()
        self.__init__(nodes, tris, self.mat, channels, fixed_dofs, self.ctl, thickness=self.thickness[parent],
                      stop_nodes=stop)
        for k, v in {**keep, **flags}.items():
            setattr(self, k, v)
        if self.pl is not None:
            self.ep_t, self.beta_t, self.alp_t = self.ep.clone(), self.beta.clone(), self.alp.clone()
        self._remap = pa

    def g(self, d_e):
        return (1-d_e)**2+self.mat.k_res

    def d_elem(self):
        return self.d[self.tri_t].mean(1)

    def strain(self, u):
        return torch.einsum("eij,ej->ei", self.B, u[self.dof_t])

    def energies(self, eps):
        if self.pl is not None:
            ee = eps-self.ep_t
            return 0.5*torch.einsum("ei,ij,ej->e", ee, self.pl.C, ee), torch.zeros_like(ee[:, 0])
        tr = eps[:, 0]+eps[:, 1]
        dd = 0.5*(eps[:, 0]-eps[:, 1])**2+0.5*eps[:, 2]**2
        if self.mat.split == "none":
            return 0.5*self.K2*tr**2+self.mu*dd, torch.zeros_like(tr)
        pos = 0.5*self.K2*torch.clamp(tr, min=0)**2+self.mu*dd
        neg = 0.5*self.K2*torch.clamp(tr, max=0)**2
        return pos, neg

    def f_fat(self, abar):
        aT = self.mat.aT()
        if self.mat.fat_deg == "log":
            r = torch.clamp(abar/aT, min=1.0)
            return torch.clamp(1.0-self.mat.kappa*torch.log10(r), min=1e-3)**2
        return torch.where(abar > aT, (2*aT/(abar+aT))**2, torch.ones_like(abar))

    def loads(self, t: float):
        f = torch.zeros(2*self.n, dtype=F64, device=DEV)
        ufix = torch.zeros(2*self.n, dtype=F64, device=DEV)
        for c, d, ref in self.ch:
            a = float(c.amplitude(np.array([t]))[0])
            if c.kind == "force":
                f.index_add_(0, d, a*ref)
            else:
                ufix[d] = a*ref
        return f, ufix

    def solve_u_plastic(self, t: float, tol: float = 1e-6, maxit: int = 30):
        f, ufix = self.loads(t)
        w = self.g(self.d_elem())*self.vol
        u = self.u.clone()
        if self.u_pred is not None:
            u, self.u_pred = self.u_pred, None
        u = torch.where(self.fixed, ufix, u)
        fn = float(torch.linalg.norm(f))

        def residual(u):
            eps = self.strain(u)
            if _j2_elem is not None:
                sig, ep, beta, alp, dg, fe, ke = _j2_elem(self.pl, eps, self.ep, self.beta, self.alp, self.B, w)
            else:
                sig, ep, beta, alp, dg = self.pl.return_map(eps, self.ep, self.beta, self.alp)
                Ct = self.pl.tangent(sig, beta, alp, dg)
                fe = torch.einsum("eki,ek->ei", self.B, sig)*w[:, None]
                ke = torch.einsum("eki,ekl,elj->eij", self.B, Ct, self.B)*w[:, None, None]
            fint = torch.zeros(2*self.n, dtype=F64, device=DEV).index_add_(0, self.dof_t.reshape(-1), fe.reshape(-1))
            R = torch.where(self.fixed, 0.0, f-fint)
            rn, fi = torch.stack([torch.linalg.norm(R), torch.linalg.norm(fint)]).tolist()
            return eps, (sig, ep, beta, alp), ke, R, rn, fi

        eps, state, ke, R, rn, fi = residual(u)
        rn_prev = None
        for it in range(maxit):
            if rn <= tol*max(fn, fi, 1e-30):
                break
            slow = rn_prev is not None and rn > self.ctl.reuse_rate*rn_prev
            if not (self.ctl.reuse_factor and self._fact_ok) or slow:
                self.Pu.assemble(ke)
                self.Pu.factor(self.fixed)
                self._fact_ok = True
                self.n_fact += 1
            du = self.Pu.back(R, self.fixed)
            step = 1.0
            for _ in range(8):
                trial = residual(u+step*du)
                if trial[4] < rn or step < 0.01:
                    break
                step *= 0.5
            u = u+step*du
            rn_prev = rn
            eps, state, ke, R, rn, fi = trial
            self.n_newton += 1
            if self.debug:
                print(f"    newton {it}: step {step:.3f} |R|/|f| {rn/max(fn, 1e-30):.3e}", flush=True)
        self.u = u
        self.sig_t, self.ep_t, self.beta_t, self.alp_t = state
        self.newton_it = it+1
        return eps, it+1

    def solve_u(self, t: float):
        if self.pl is not None:
            return self.solve_u_plastic(t)
        f, ufix = self.loads(t)
        ge = self.g(self.d_elem())
        if self.KD_e is None:
            self.KD_e = torch.einsum("eki,kl,elj->eij", self.B, self.CD, self.B)*self.vol[:, None, None]
            self.KI_e = torch.einsum("eki,kl,elj->eij", self.B, self.CI, self.B)*self.vol[:, None, None]
        n_it = 1 if self.ctl.lag_mask else self.ctl.max_mask_iter
        for it in range(n_it):
            if self.mat.split == "none":
                ke = ge[:, None, None]*(self.KD_e+self.KI_e)
            else:
                wI = torch.where(self.comp, 1.0, ge)
                ke = ge[:, None, None]*self.KD_e+wI[:, None, None]*self.KI_e
            self.Pu.assemble(ke)
            if self.has_disp:
                self.u = self.Pu.solve(f, self.fixed, ufix)
            else:
                self.Pu.factor(self.fixed)
                self.u = self.Pu.back(f, self.fixed)
            eps = self.strain(self.u)
            if self.mat.split == "none":
                self.mask_changed = False
                return eps, it+1
            tr = eps[:, 0]+eps[:, 1]
            comp = tr < 0
            self.mask_changed = bool(((comp != self.comp) & (tr.abs() > 1e-9)).any())
            self.comp = comp
            if not self.mask_changed:
                return eps, it+1
        return eps, n_it

    def solve_d(self):
        Gc = self.mat.Gc*self.f_fat(self.abar)
        l = self.mat.ell
        GG = self.GG if self.A_el is None else torch.einsum("eia,eab,ejb->eij", self.G, self.A_el, self.G)
        ke = ((Gc*l)[:, None, None]*GG
              + (Gc/l+2*self.H)[:, None, None]*self.Me)*self.vol[:, None, None]
        self.Pd.assemble(ke)
        fe = (2*self.H*self.vol/3.0)[:, None].expand(-1, 3)
        rhs = torch.zeros(self.n, dtype=F64, device=DEV).index_add_(0, self.tri_t.reshape(-1), fe.reshape(-1))
        d = self.Pd.solve(rhs)
        return torch.maximum(d.clamp(0.0, 1.0), self.d)

    def substep(self, t: float):
        d_old = self.d.clone()
        H_old = self.H.clone()
        if self.pl is not None:
            if self.u_prev is not None and t > 0:
                self.u_pred = 2*self.u-self.u_prev
            self.u_prev = self.u.clone() if t > 0 else None
        for k in range(self.ctl.max_stagger):
            eps, nmask = self.solve_u(t)
            pos, _ = self.energies(eps)
            self.H = torch.maximum(H_old, pos)
            d_new = self.solve_d()
            err = float((d_new-self.d).abs().max())
            if self.ctl.omega != 1.0:
                d_new = (self.d+self.ctl.omega*(d_new-self.d)).clamp(max=1.0)
            self.d = d_new
            mask_open = self.ctl.lag_mask and self.mask_changed and not (0 < self.ctl.mask_cap <= k+1)
            if err < self.ctl.stagger_tol and not mask_open:
                break
        eps = self.strain(self.u)
        if self.pl is not None:
            dWp = (self.sig_t*(self.ep_t-self.ep)).sum(1).clamp(min=0)
            self.Wp += dWp
            if self.record_u:
                self.dep_hist.append(self.ep_t-self.ep)
            self.ep, self.beta, self.alp = self.ep_t, self.beta_t, self.alp_t
        if self.mat.fatigue == "chord":
            self.eps_hist.append(eps)
        else:
            pos, _ = self.energies(eps)
            ge = self.g(self.d_elem())
            alpha = ge*pos
            if self.mat.sigma_e > 0:
                alpha = torch.clamp(alpha-self.mat.sigma_e**2/(2*self.mat.E), min=0)
            self.abar += torch.clamp(alpha-self.alpha_prev, min=0)
            if self.pl is not None:
                self.abar += self.mat.chi_p*ge*dWp
            self.alpha_prev = alpha
        return k+1, float((self.d-d_old).abs().max())

    def chord_energy(self) -> torch.Tensor:
        E = torch.stack(self.eps_hist)
        tr = E[:, :, 0]+E[:, :, 1]
        best = torch.zeros(self.m, dtype=F64, device=DEV)
        for i in range(len(E)-1):
            de = E[i+1:]-E[i]
            dtr = de[:, :, 0]+de[:, :, 1]
            dev = 0.5*(de[:, :, 0]-de[:, :, 1])**2+0.5*de[:, :, 2]**2
            open_ = (tr[i+1:] > 0) | (tr[i] > 0)
            q = 0.5*self.K2*dtr**2*open_+self.mat.w_shear*self.mu*dev
            best = torch.maximum(best, q.max(0).values)
        return best

    def resolved_cycle(self):
        K = self.ctl.substeps
        iters = 0
        self.eps_hist = []
        self.u_hist, self.dep_hist = [], []
        for i in range(K):
            it, _ = self.substep((i+1)/K)
            iters += it
            if self.record_u:
                self.u_hist.append(self.u.clone())
        if self.mat.fatigue == "chord":
            self.abar += self.g(self.d_elem())*self.chord_energy()
            self.eps_hist = []
        return iters

    def run(self, hooks: Callable | None = None, restart: str | None = None):
        out = Path(self.ctl.out)
        out.mkdir(parents=True, exist_ok=True)
        rows, t0 = [], time.time()
        if restart:
            N, n_res, dN = self.load(restart)
            if (out/"history.csv").exists():
                with open(out/"history.csv") as fh:
                    rows = [r for r in csv.DictReader(fh) if int(r["resolved"]) <= n_res]
            print(f"restart from {restart}: N={N}, resolved={n_res}", flush=True)
        else:
            N, n_res, dN = 0, 0, self.ctl.dN_init
            self.substep(0.0)
        while N < self.ctl.n_cycles:
            self._ck = self.checkpoint() if self.keep_checkpoint else None
            d0, ab0 = self.d.clone(), self.abar.clone()
            it = self.resolved_cycle()
            n_res += 1
            dd = float((self.d-d0).abs().max())
            if os.environ.get("PFF_DEBUG_DD"):
                i = int((self.d-d0).abs().argmax())
                print(f"  dd at node {i} {self.nodes[i].tolist()} d {float(d0[i]):.3f} -> {float(self.d[i]):.3f}",
                      flush=True)
            dab = self.abar-ab0
            if dN > 1:
                self.abar += (dN-1)*dab
            N += dN
            row = dict(N=N, dN=dN, resolved=n_res, d_max=float(self.d.max()), dd_cycle=dd,
                       abar_max=float(self.abar.max()), f_min=float(self.f_fat(self.abar).min()),
                       stagger=it, wall=time.time()-t0)
            if hooks is not None:
                row.update(hooks(self))
            if self.redo:
                self.redo = False
                N -= dN
                n_res -= 1
                dN = max(1, dN//2)
                print(f"  -> cycle redone on the refined mesh ({self.m} elements), dN={dN}", flush=True)
                continue
            if self._remap is not None:
                dab, self._remap = dab[self._remap], None
            row["elements"] = self.m
            rows.append(row)
            if n_res % self.ctl.save_every == 0 or n_res == 1:
                self.snapshot(out/f"snap_{n_res:05d}.npz", N, n_res, dN)
                self.write_rows(out/"history.csv", rows)
            print(f"N={N:8d} dN={dN:5d} dmax={row['d_max']:.3f} dd={dd:.2e} fmin={row['f_min']:.3f} "
                  f"it={it:4d} wall={row['wall']:.0f}s", flush=True)
            if self.stop_nodes is not None and float(self.d[self.stop_nodes].max()) >= self.ctl.d_stop:
                break
            if (out/"STOP").exists():
                self.snapshot(out/f"snap_{n_res:05d}.npz", N, n_res, dN)
                self.write_rows(out/"history.csv", rows)
                (out/"STOP").unlink()
                print(f"stopped by STOP file at N={N}, snapshot snap_{n_res:05d}.npz", flush=True)
                break
            fa = self.f_fat(self.abar)
            cand = [self.ctl.dN_max, max(1, 2*dN)]
            if dd > 0:
                cand.append(max(1, int(self.ctl.dd_target/dd)))
            lo, hi = 1, int(min(cand))
            while lo < hi:
                mid = (lo+hi+1)//2
                df = float((self.f_fat(self.abar+mid*dab)-fa).abs().max())
                if df <= self.ctl.df_target:
                    lo = mid
                else:
                    hi = mid-1
            dN = lo
        self.snapshot(out/f"snap_{n_res:05d}.npz", N, n_res, dN)
        self.write_rows(out/"history.csv", rows)
        return rows

    _CK = ("d", "u", "u_prev", "H", "abar", "alpha_prev", "comp", "A_el", "ep", "beta", "alp", "Wp",
           "ep_t", "beta_t", "alp_t", "sig_t")

    def checkpoint(self) -> dict:
        return {k: getattr(self, k).clone() for k in self._CK if getattr(self, k, None) is not None}

    def restore(self, ck: dict):
        for k, v in ck.items():
            setattr(self, k, v.clone())
        self._fact_ok = False

    def elastic_cycle(self, K: int) -> torch.Tensor:
        ck = self.checkpoint()
        U = []
        for i in range(K):
            self.solve_u((i+1)/K)
            U.append(self.u.clone())
        self.restore(ck)
        self.mask_changed = False
        return torch.stack(U)

    def snapshot(self, path: Path, N: int, n_res: int = 0, dN: int = 1):
        extra = {}
        if self.pl is not None:
            extra = {k: getattr(self, k).cpu().numpy() for k in ("ep", "beta", "alp", "Wp")}
        if self.A_el is not None:
            extra["A_el"] = self.A_el.cpu().numpy()
        if self.n_remesh:
            extra.update(nodes=self.nodes, tris=self.tris, thickness=self.thickness)
        np.savez_compressed(path, N=N, n_res=n_res, dN=dN, d=self.d.cpu().numpy(), u=self.u.cpu().numpy(),
                            abar=self.abar.cpu().numpy(), H=self.H.cpu().numpy(),
                            alpha_prev=self.alpha_prev.cpu().numpy(), comp=self.comp.cpu().numpy(), **extra)

    def load(self, path: str) -> tuple[int, int, int]:
        z = np.load(path)
        self.d, self.u, self.abar, self.H = T_(z["d"]), T_(z["u"]), T_(z["abar"]), T_(z["H"])
        self.alpha_prev = T_(z["alpha_prev"])
        self.comp = T_(z["comp"], torch.bool)
        if self.pl is not None:
            for k in ("ep", "beta", "alp", "Wp"):
                setattr(self, k, T_(z[k]))
        if "A_el" in z:
            self.A_el = T_(z["A_el"])
        return int(z["N"]), int(z["n_res"]), int(z["dN"])

    @staticmethod
    def write_rows(path: Path, rows: list[dict]):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval="")
            w.writeheader()
            w.writerows(rows)
