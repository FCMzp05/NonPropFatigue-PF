from __future__ import annotations

import time

import numpy as np
import torch

from .solver import DEV, F64, T_, FatigueSolver, Pattern


class FatigueSolver3D(FatigueSolver):
    def __init__(self, nodes: np.ndarray, tets: np.ndarray, mat, channels, fixed_dofs: np.ndarray, ctl,
                 d0: np.ndarray | None = None, stop_nodes: np.ndarray | None = None):
        if mat.sy0 is not None or mat.fatigue != "carrara":
            raise ValueError("3D solver: elastic Carrara fatigue only")
        self.mat, self.ctl, self.channels = mat, ctl, channels
        self.nodes, self.tris = nodes, tets
        self.thickness = np.ones(len(tets))
        self.n_remesh, self._remap = 0, None
        self.keep_checkpoint, self.redo, self._ck = False, False, None
        n, m = len(nodes), len(tets)
        self.n, self.m = n, m
        x = nodes[tets]
        J = np.stack([x[:, 1]-x[:, 0], x[:, 2]-x[:, 0], x[:, 3]-x[:, 0]], axis=2)
        det = np.linalg.det(J)
        if np.any(det <= 0):
            raise ValueError("non-positive element orientation")
        dN = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        G = dN@np.linalg.inv(J)
        self.vol = T_(det/6.0)
        self.area = self.vol
        self.G = T_(G)
        B = np.zeros((m, 6, 12))
        for a in range(3):
            B[:, a, a::3] = G[:, :, a]
        B[:, 3, 1::3], B[:, 3, 2::3] = G[:, :, 2], G[:, :, 1]
        B[:, 4, 0::3], B[:, 4, 2::3] = G[:, :, 2], G[:, :, 0]
        B[:, 5, 0::3], B[:, 5, 1::3] = G[:, :, 1], G[:, :, 0]
        self.B = T_(B)
        self.tri_t = T_(tets, torch.int64)
        self.dof_t = (self.tri_t[:, :, None]*3+torch.arange(3, device=DEV)).reshape(m, 12)
        t0 = time.time()
        self.Pu = Pattern(tets, n, 3)
        self.Pd = Pattern(tets, n, 1)
        self.t_setup = time.time()-t0
        E, nu = mat.E, mat.nu
        mu = E/(2*(1+nu))
        lam = E*nu/((1+nu)*(1-2*nu))
        self.K2, self.mu = lam+2*mu/3, mu
        one = torch.zeros(6, 6, dtype=F64, device=DEV)
        one[:3, :3] = 1.0
        dev = torch.zeros(6, 6, dtype=F64, device=DEV)
        dev[:3, :3] = 2*mu*(torch.eye(3, dtype=F64, device=DEV)-1.0/3.0)
        dev[3:, 3:] = mu*torch.eye(3, dtype=F64, device=DEV)
        self.CI, self.CD = self.K2*one, dev
        self.fixed = torch.zeros(3*n, dtype=torch.bool, device=DEV)
        self.fixed[T_(fixed_dofs, torch.int64)] = True
        self.ch = []
        for c in channels:
            d = T_(c.dofs, torch.int64)
            if c.kind == "disp":
                self.fixed[d] = True
            self.ch.append((c, d, T_(c.ref)))
        self.u = torch.zeros(3*n, dtype=F64, device=DEV)
        self.d = torch.zeros(n, dtype=F64, device=DEV) if d0 is None else T_(d0)
        self.H = torch.zeros(m, dtype=F64, device=DEV)
        self.abar = torch.zeros(m, dtype=F64, device=DEV)
        self.alpha_prev = torch.zeros(m, dtype=F64, device=DEV)
        self.comp = torch.zeros(m, dtype=torch.bool, device=DEV)
        self.eps_hist = []
        self.pl = None
        self.debug = False
        self.u_pred, self.u_prev = None, None
        self._fact_ok, self.n_fact, self.n_newton = False, 0, 0
        self.A_el = None
        self.record_u, self.u_hist, self.dep_hist = False, [], []
        self.KD_e = self.KI_e = None
        self.mask_changed = False
        self.has_disp = any(c.kind == "disp" for c in channels)
        self.stop_nodes = None if stop_nodes is None else T_(stop_nodes, torch.int64)
        self.Me = (torch.ones(4, 4, dtype=F64, device=DEV)+torch.eye(4, dtype=F64, device=DEV))/20.0
        self.GG = torch.einsum("eia,eja->eij", self.G, self.G)

    def remesh(self, nodes: np.ndarray, tets: np.ndarray, parent: np.ndarray, mids: list[tuple],
               channels, fixed_dofs: np.ndarray, nodal: tuple | None = None, low: tuple | None = None):
        pa = T_(parent, torch.int64)
        passes = [(T_(ea, torch.int64), T_(eb, torch.int64)) for ea, eb in mids]
        if nodal is not None:
            ni, nw = T_(nodal[0], torch.int64), T_(nodal[1], F64)

        def nod(x):
            if nodal is not None:
                return (x[ni]*(nw if x.dim() == 1 else nw[:, :, None])).sum(1)
            for ia, ib in passes:
                x = torch.cat([x, 0.5*(x[ia]+x[ib])])
            return x

        H, abar = self.H[pa], self.abar[pa]
        if low is not None:
            li, lm = T_(low[0], torch.int64), T_(low[1], torch.bool)
            H = torch.where(lm, self.H[li].min(1).values, H)
            abar = torch.where(lm, self.abar[li].min(1).values, abar)
        keep = dict(d=nod(self.d), u=nod(self.u.view(-1, 3)).reshape(-1),
                    u_prev=None if self.u_prev is None else nod(self.u_prev.view(-1, 3)).reshape(-1),
                    H=H, abar=abar, alpha_prev=self.alpha_prev[pa], comp=self.comp[pa],
                    A_el=None if self.A_el is None else self.A_el[pa])
        flags = dict(record_u=self.record_u, debug=self.debug, n_fact=self.n_fact, n_newton=self.n_newton,
                     n_remesh=self.n_remesh+1)
        stop = None if self.stop_nodes is None else self.stop_nodes.cpu().numpy()
        self.Pu.free()
        self.Pd.free()
        self.__init__(nodes, tets, self.mat, channels, fixed_dofs, self.ctl, stop_nodes=stop)
        for k, v in {**keep, **flags}.items():
            setattr(self, k, v)
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
        self._remap = pa

    def energies(self, eps):
        tr = eps[:, 0]+eps[:, 1]+eps[:, 2]
        m_ = tr/3.0
        dd = ((eps[:, :3]-m_[:, None])**2).sum(1)+0.5*(eps[:, 3:]**2).sum(1)
        if self.mat.split == "none":
            return 0.5*self.K2*tr**2+self.mu*dd, torch.zeros_like(tr)
        return 0.5*self.K2*torch.clamp(tr, min=0)**2+self.mu*dd, 0.5*self.K2*torch.clamp(tr, max=0)**2

    def loads(self, t: float):
        f = torch.zeros(3*self.n, dtype=F64, device=DEV)
        ufix = torch.zeros(3*self.n, dtype=F64, device=DEV)
        for c, d, ref in self.ch:
            a = float(c.amplitude(np.array([t]))[0])
            if c.kind == "force":
                f.index_add_(0, d, a*ref)
            else:
                ufix[d] = a*ref
        return f, ufix

    def solve_u(self, t: float):
        f, ufix = self.loads(t)
        ge = self.g(self.d_elem())
        if self.KD_e is None:
            self.KD_e = torch.einsum("eki,kl,elj->eij", self.B, self.CD, self.B)*self.vol[:, None, None]
            self.KI_e = torch.einsum("eki,kl,elj->eij", self.B, self.CI, self.B)*self.vol[:, None, None]
        n_it = 1 if self.ctl.lag_mask else self.ctl.max_mask_iter
        for it in range(n_it):
            wI = ge if self.mat.split == "none" else torch.where(self.comp, 1.0, ge)
            self.Pu.assemble(ge[:, None, None]*self.KD_e+wI[:, None, None]*self.KI_e)
            if self.has_disp:
                self.u = self.Pu.solve(f, self.fixed, ufix)
            else:
                x = None
                if self.ctl.pcg_tol > 0:
                    self.Pu.factor32(self.fixed)
                    x, k = self.Pu.pcg(f, self.fixed, self.u, self.ctl.pcg_tol, self.ctl.pcg_maxit, self.Pu.back32)
                if x is None:
                    self.Pu.factor(self.fixed)
                    x = self.Pu.back(f, self.fixed)
                self.u = x
            eps = self.strain(self.u)
            if self.mat.split == "none":
                self.mask_changed = False
                return eps, it+1
            tr = eps[:, 0]+eps[:, 1]+eps[:, 2]
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
        ke = ((Gc*l)[:, None, None]*GG+(Gc/l+2*self.H)[:, None, None]*self.Me)*self.vol[:, None, None]
        self.Pd.assemble(ke)
        fe = (2*self.H*self.vol/4.0)[:, None].expand(-1, 4)
        rhs = torch.zeros(self.n, dtype=F64, device=DEV).index_add_(0, self.tri_t.reshape(-1), fe.reshape(-1))
        d = self.Pd.solve(rhs)
        return torch.maximum(d.clamp(0.0, 1.0), self.d)
