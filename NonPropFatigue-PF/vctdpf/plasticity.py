from __future__ import annotations

import math
from dataclasses import dataclass

import torch

F64 = torch.float64
S2 = math.sqrt(2.0)


@dataclass
class J2PlaneStress:
    E: float
    nu: float
    sy0: float
    H_iso: float = 0.0
    H_kin: float = 0.0

    def __post_init__(self):
        E, nu = self.E, self.nu
        self.mu = E/(2*(1+nu))
        self.C = torch.tensor([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]], dtype=F64)*E/(1-nu**2)
        hk = 2.0/3.0*self.H_kin
        self.lam = (E/(1-nu)/3+hk, 2*self.mu+hk, 2*self.mu+hk)
        self.p = (1.0/3.0, 1.0, 2.0)

    def to(self, dev):
        self.C = self.C.to(dev)
        return self

    def kappa(self, a):
        return self.sy0+self.H_iso*a

    @staticmethod
    def Peta(eta):
        return torch.stack([(2*eta[:, 0]-eta[:, 1])/3, (2*eta[:, 1]-eta[:, 0])/3, 2*eta[:, 2]], 1)

    def return_map(self, eps, ep_n, beta_n, alpha_n, iters: int = 15):
        sig_tr = (eps-ep_n)@self.C.T
        eta_tr = sig_tr-beta_n
        a1 = (eta_tr[:, 0]+eta_tr[:, 1])/S2
        a2 = (eta_tr[:, 0]-eta_tr[:, 1])/S2
        a3 = eta_tr[:, 2]
        A = (a1, a2, a3)
        p, lam = self.p, self.lam
        f2_tr = p[0]*a1**2+p[1]*a2**2+p[2]*a3**2
        k_tr = self.kappa(alpha_n)
        plastic = 0.5*f2_tr-k_tr**2/3 > 1e-12*k_tr**2
        c23 = math.sqrt(2.0/3.0)
        fb_tr = torch.sqrt(f2_tr.clamp(min=1e-300))
        l2 = lam[1]
        y = (c23*k_tr+c23**2*self.H_iso*fb_tr/l2)/(1+c23**2*self.H_iso/l2)
        dg = torch.where(plastic, ((fb_tr/y-1)/l2).clamp(min=0), torch.zeros_like(a1))
        for _ in range(iters):
            den = [1+dg*l for l in lam]
            f2 = sum(p[i]*A[i]**2/den[i]**2 for i in range(3))
            df2 = sum(-2*p[i]*A[i]**2*lam[i]/den[i]**3 for i in range(3))
            fb = torch.sqrt(f2.clamp(min=1e-300))
            k = self.kappa(alpha_n+c23*dg*fb)
            phi = 0.5*f2-k**2/3
            dfb = df2/(2*fb)
            dphi = 0.5*df2-2.0/3.0*k*self.H_iso*c23*(fb+dg*dfb)
            dg = torch.where(plastic, (dg-phi/dphi).clamp(min=0), dg)
        den = [1+dg*l for l in lam]
        b1, b2, b3 = A[0]/den[0], A[1]/den[1], A[2]/den[2]
        eta = torch.stack([(b1+b2)/S2, (b1-b2)/S2, b3], 1)
        Pe = self.Peta(eta)
        dep = dg[:, None]*Pe
        ep = ep_n+dep
        beta = beta_n+2.0/3.0*self.H_kin*dg[:, None]*eta
        f2 = p[0]*b1**2+p[1]*b2**2+p[2]*b3**2
        alpha = alpha_n+math.sqrt(2.0/3.0)*dg*torch.sqrt(f2.clamp(min=0))
        sig = beta+eta
        return sig, ep, beta, alpha, dg

    def tangent(self, sig, beta, alpha, dg):
        dev, m = sig.device, sig.shape[0]
        if not hasattr(self, "_Q") or self._Q.device != dev:
            self._Q = torch.tensor([[1/S2, 1/S2, 0], [1/S2, -1/S2, 0], [0, 0, 1]], dtype=F64, device=dev)
            self._c = torch.tensor([self.E/(1-self.nu), 2*self.mu, self.mu], dtype=F64, device=dev)
            self._p = torch.tensor(self.p, dtype=F64, device=dev)
            self._lam = torch.tensor(self.lam, dtype=F64, device=dev)
        Q, c, p, lam = self._Q, self._c, self._p, self._lam
        eta = sig-beta
        a = eta@Q
        fb = torch.sqrt((p*a**2).sum(1).clamp(min=1e-300))
        k = self.kappa(alpha)
        at_yield = (dg > 0) | (fb >= math.sqrt(2.0/3.0)*k*(1-1e-6))
        hk = 2.0/3.0*self.H_kin
        Ai = 1/(1+dg[:, None]*lam)
        c23 = math.sqrt(2.0/3.0)
        th1 = 1-2.0/3.0*k*self.H_iso*c23*dg/fb
        th2 = 2.0/3.0*k*self.H_iso*c23*fb
        z = Ai*c*p*a
        den = th2+th1*(p*lam*Ai*a**2).sum(1)
        Ce = torch.diag_embed((1+hk*dg)[:, None]*Ai*c)-(th1/den)[:, None, None]*z[:, :, None]*z[:, None, :]
        Cp = Q@Ce@Q.T
        return torch.where(at_yield[:, None, None], Cp, self.C.expand(m, 3, 3))
