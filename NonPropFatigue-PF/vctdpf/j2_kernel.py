from __future__ import annotations

import cupy as cp
import torch

_SRC = r"""
extern "C" __global__ void j2ps(const double* __restrict__ eps, const double* __restrict__ epn,
                                const double* __restrict__ betan, const double* __restrict__ alphan,
                                double* sig, double* ep, double* beta, double* alpha, double* dgo, double* Ct,
                                const int m, const double E, const double nu, const double sy0,
                                const double Hi, const double Hk,
                                const double* __restrict__ B, const double* __restrict__ w,
                                double* fe, double* ke)
{
    const int e = blockIdx.x*blockDim.x+threadIdx.x;
    if (e >= m) return;
    const double r2 = 0.70710678118654752440, c23 = 0.81649658092772603273;
    const double mu = E/(2.0*(1.0+nu)), c0 = E/(1.0-nu*nu);
    const double hk = 2.0/3.0*Hk;
    const double cc[3] = {E/(1.0-nu), 2.0*mu, mu};
    const double pp[3] = {1.0/3.0, 1.0, 2.0};
    const double ll[3] = {cc[0]*pp[0]+hk, cc[1]*pp[1]+hk, cc[2]*pp[2]+hk};
    const int i3 = 3*e;
    const double ee0 = eps[i3]-epn[i3], ee1 = eps[i3+1]-epn[i3+1], ee2 = eps[i3+2]-epn[i3+2];
    const double s0 = c0*(ee0+nu*ee1), s1 = c0*(nu*ee0+ee1), s2 = mu*ee2;
    const double t0 = s0-betan[i3], t1 = s1-betan[i3+1], t2 = s2-betan[i3+2];
    const double a[3] = {(t0+t1)*r2, (t0-t1)*r2, t2};
    const double an = alphan[e];
    const double f2tr = pp[0]*a[0]*a[0]+pp[1]*a[1]*a[1]+pp[2]*a[2]*a[2];
    const double ktr = sy0+Hi*an;
    double dg = 0.0;
    if (0.5*f2tr-ktr*ktr/3.0 > 1e-12*ktr*ktr) {
        const double fbtr = sqrt(f2tr), l2 = ll[1];
        const double y = (c23*ktr+c23*c23*Hi*fbtr/l2)/(1.0+c23*c23*Hi/l2);
        dg = fmax((fbtr/y-1.0)/l2, 0.0);
        for (int it = 0; it < 40; ++it) {
            double f2 = 0.0, df2 = 0.0;
            for (int i = 0; i < 3; ++i) {
                const double d = 1.0+dg*ll[i];
                f2 += pp[i]*a[i]*a[i]/(d*d);
                df2 += -2.0*pp[i]*a[i]*a[i]*ll[i]/(d*d*d);
            }
            const double fb = sqrt(fmax(f2, 1e-300));
            const double k = sy0+Hi*(an+c23*dg*fb);
            const double phi = 0.5*f2-k*k/3.0;
            const double dphi = 0.5*df2-2.0/3.0*k*Hi*c23*(fb+dg*df2/(2.0*fb));
            const double step = phi/dphi;
            dg = fmax(dg-step, 0.0);
            if (fabs(step) <= 1e-15*fmax(dg, 1e-300)) break;
        }
    }
    double b[3], A[3];
    for (int i = 0; i < 3; ++i) { A[i] = 1.0/(1.0+dg*ll[i]); b[i] = a[i]*A[i]; }
    const double h0 = (b[0]+b[1])*r2, h1 = (b[0]-b[1])*r2, h2 = b[2];
    const double P0 = (2.0*h0-h1)/3.0, P1 = (2.0*h1-h0)/3.0, P2 = 2.0*h2;
    ep[i3] = epn[i3]+dg*P0; ep[i3+1] = epn[i3+1]+dg*P1; ep[i3+2] = epn[i3+2]+dg*P2;
    const double bb0 = betan[i3]+hk*dg*h0, bb1 = betan[i3+1]+hk*dg*h1, bb2 = betan[i3+2]+hk*dg*h2;
    beta[i3] = bb0; beta[i3+1] = bb1; beta[i3+2] = bb2;
    sig[i3] = bb0+h0; sig[i3+1] = bb1+h1; sig[i3+2] = bb2+h2;
    const double f2 = pp[0]*b[0]*b[0]+pp[1]*b[1]*b[1]+pp[2]*b[2]*b[2];
    const double fb = sqrt(fmax(f2, 1e-300));
    const double al = an+c23*dg*fb;
    alpha[e] = al;
    dgo[e] = dg;
    // consistent tangent
    double T[9];
    const double k = sy0+Hi*al;
    if (!(dg > 0.0 || fb >= c23*k*(1.0-1e-6))) {
        T[0] = c0; T[1] = c0*nu; T[2] = 0.0;
        T[3] = c0*nu; T[4] = c0; T[5] = 0.0;
        T[6] = 0.0; T[7] = 0.0; T[8] = mu;
    } else {
        const double th1 = 1.0-2.0/3.0*k*Hi*c23*dg/fb, th2 = 2.0/3.0*k*Hi*c23*fb;
        double z[3], den = th2;
        for (int i = 0; i < 3; ++i) { z[i] = A[i]*cc[i]*pp[i]*b[i]; den += th1*pp[i]*ll[i]*A[i]*b[i]*b[i]; }
        double Ce[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                Ce[i][j] = (i == j ? (1.0+hk*dg)*A[i]*cc[i] : 0.0)-th1*z[i]*z[j]/den;
        const double Q[3][3] = {{r2, r2, 0.0}, {r2, -r2, 0.0}, {0.0, 0.0, 1.0}};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double s = 0.0;
                for (int p = 0; p < 3; ++p)
                    for (int q = 0; q < 3; ++q)
                        s += Q[i][p]*Ce[p][q]*Q[j][q];
                T[3*i+j] = s;
            }
    }
    if (Ct) for (int i = 0; i < 9; ++i) Ct[9*e+i] = T[i];
    if (!B) return;
    // element internal force w B^T sigma and stiffness w B^T C B, w = g(d) * volume
    const double* Be = B+18*e;
    const double we = w[e], sg[3] = {sig[i3], sig[i3+1], sig[i3+2]};
    double CB[3][6];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 6; ++j)
            CB[i][j] = T[3*i]*Be[j]+T[3*i+1]*Be[6+j]+T[3*i+2]*Be[12+j];
    for (int a = 0; a < 6; ++a) {
        fe[6*e+a] = we*(Be[a]*sg[0]+Be[6+a]*sg[1]+Be[12+a]*sg[2]);
        for (int j = 0; j < 6; ++j)
            ke[36*e+6*a+j] = we*(Be[a]*CB[0][j]+Be[6+a]*CB[1][j]+Be[12+a]*CB[2][j]);
    }
}
"""

_kernel = cp.RawKernel(_SRC, "j2ps")


def _cp(t: torch.Tensor):
    return cp.from_dlpack(t.contiguous())


def _launch(mat, eps, ep_n, beta_n, alpha_n, Ct, B, w, fe, ke):
    m = eps.shape[0]
    opt = dict(dtype=eps.dtype, device=eps.device)
    sig, ep, beta = (torch.empty(m, 3, **opt) for _ in range(3))
    alpha, dg = torch.empty(m, **opt), torch.empty(m, **opt)
    ptr = lambda t: cp.uint64(0) if t is None else _cp(t)
    threads = 128
    _kernel(((m+threads-1)//threads,), (threads,),
            (_cp(eps), _cp(ep_n), _cp(beta_n), _cp(alpha_n), _cp(sig), _cp(ep), _cp(beta), _cp(alpha), _cp(dg),
             ptr(Ct), cp.int32(m), cp.float64(mat.E), cp.float64(mat.nu), cp.float64(mat.sy0),
             cp.float64(mat.H_iso), cp.float64(mat.H_kin), ptr(B), ptr(w), ptr(fe), ptr(ke)))
    return sig, ep, beta, alpha, dg


def return_map_tangent(mat, eps, ep_n, beta_n, alpha_n):
    Ct = torch.empty(eps.shape[0], 3, 3, dtype=eps.dtype, device=eps.device)
    return (*_launch(mat, eps, ep_n, beta_n, alpha_n, Ct, None, None, None, None), Ct)


def return_map_element(mat, eps, ep_n, beta_n, alpha_n, B, w):
    m = eps.shape[0]
    fe = torch.empty(m, 6, dtype=eps.dtype, device=eps.device)
    ke = torch.empty(m, 6, 6, dtype=eps.dtype, device=eps.device)
    return (*_launch(mat, eps, ep_n, beta_n, alpha_n, None, B.contiguous(), w.contiguous(), fe, ke), fe, ke)
