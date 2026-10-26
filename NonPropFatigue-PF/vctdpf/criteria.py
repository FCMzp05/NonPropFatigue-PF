from __future__ import annotations

import sys

import numpy as np

TH = np.linspace(-np.pi*0.95, np.pi*0.95, 3801)


def resample(t, KI, KII, n=720):
    A = np.c_[np.ones_like(t), np.cos(2*np.pi*t), np.sin(2*np.pi*t)]
    tt = np.linspace(0, 1, n, endpoint=False)
    At = np.c_[np.ones_like(tt), np.cos(2*np.pi*tt), np.sin(2*np.pi*tt)]
    fit = lambda K: At@np.linalg.lstsq(A, K, rcond=None)[0]
    return tt, fit(KI), fit(KII)


def kink(KI, KII, th=TH):
    c1, c3 = np.cos(th/2), np.cos(3*th/2)
    s1, s3 = np.sin(th/2), np.sin(3*th/2)
    KI, KII = KI[:, None], KII[:, None]
    k1 = 0.25*KI*(3*c1+c3)-0.75*KII*(s1+s3)
    k2 = 0.25*KI*(s1+s3)+0.25*KII*(c1+3*c3)
    return k1, k2


def local_ext(f, th=TH, kind="max", rel=0.01):
    i = np.arange(1, len(f)-1)
    m = (f[i] >= f[i-1]) & (f[i] >= f[i+1]) if kind == "max" else (f[i] <= f[i-1]) & (f[i] <= f[i+1])
    idx = i[m]
    if len(idx) == 0:
        return []
    best = f[idx].max() if kind == "max" else f[idx].min()
    ok = np.abs(f[idx]-best) <= rel*abs(best)
    return sorted(th[idx[ok]])


def mts(KI, KII, w=0.5):
    k1, _ = kink(KI, KII)
    k1p = np.clip(k1, 0, None)
    kmax, dk = k1p.max(0), k1p.max(0)-k1p.min(0)
    return local_ext(kmax**(1-w)*dk**w)


def hp(KI, KII, c=0.7):
    k1, k2 = kink(KI, KII)
    k1p = np.clip(k1, 0, None)
    return local_ext(np.hypot(k1p.max(0)-k1p.min(0), c*(k2.max(0)-k2.min(0))))


def mssr(KI, KII):
    _, k2 = kink(KI, KII)
    return local_ext(k2.max(0)-k2.min(0))


def mpzr(KI, KII):
    th = TH[None, :]
    a, b = KI[:, None], KII[:, None]
    s2 = np.sin(th/2)**2
    rp = (a**2*np.cos(th/2)**2*(1+3*s2)+a*b*np.sin(th)*(3*np.cos(th)-1)
          + b**2*(3+s2*(1-9*np.cos(th/2)**2)))
    m = np.abs(TH) < np.pi/2
    return local_ext(np.where(m, rp.max(0), np.inf), kind="min")


def upsample(x, n):
    m = len(x)
    if m >= n:
        return x
    s = np.linspace(0, m, n, endpoint=False)
    return np.interp(s, np.arange(m+1), np.r_[x, x[0]])


def vctd(KI, KII, star=True, mu=np.inf, n_up=256, rule="max", opening="mean", q=0.5):
    KI, KII = upsample(np.asarray(KI, float), n_up), upsample(np.asarray(KII, float), n_up)
    dI = (KI if opening == "contact" else KI-(KI.min()+q*(KI.max()-KI.min())) if opening == "level"
          else KI-0.5*(KI.max()+KI.min()))
    dII = KII-0.5*(KII.max()+KII.min())
    A = (np.clip(dI, 0, None) if opening in ("contact", "level") else np.where(KI > 0, dI, 0.0) if star
         else np.clip(dI, 0, None))
    d = np.sqrt(np.clip(A**2+2*A*np.abs(dII)+2*dII**2, 0, None))
    th = np.arcsin(np.clip(np.divide(dII, d, out=np.zeros_like(d), where=d > 0), -1, 1))
    e = np.c_[np.cos(th), np.sin(th)]
    dd = np.clip(np.diff(np.r_[d, d[0]]), 0, None)
    if rule == "sum":
        v = (dd[:, None]*e).sum(0)
        return [float(np.arctan2(v[1], v[0]))]
    lock = KI/np.maximum(np.abs(dII), 1e-30) >= mu
    out = []
    for fpm in ((dII >= 0) | lock, (dII <= 0) | lock):
        out.append((dd*fpm)[:, None]*e)
    da = [o.sum(0) for o in out]
    v = max(da, key=np.linalg.norm)
    return [float(np.arctan2(v[1], v[0]))]


def vctd_odd(KI, KII, **kw):
    KI, KII = np.asarray(KI, float), np.asarray(KII, float)
    f = vctd(KI, KII, **kw)[0]
    r = vctd(KI[::-1], KII[::-1], **kw)[0]
    return [0.5*float(np.angle(np.exp(1j*(f-r))))]


def all_criteria(t, KI, KII):
    t, KI, KII = resample(t, KI, KII)
    res = {}
    for w in (0.1, 0.5, 0.9):
        res[f"MTS w={w}"] = mts(KI, KII, w)
    for c in (0.7, 0.9, 1.2):
        res[f"HP c={c}"] = hp(KI, KII, c)
    res["MPZR"] = mpzr(KI, KII)
    res["MSSR"] = mssr(KI, KII)
    res["VCTD"] = vctd(KI, KII, star=False)
    for mu in (np.inf, 2.0, 1.0, 0.0):
        res[f"VCTD* mu={mu:g}"] = vctd(KI, KII, star=True, mu=mu)
    return res


if __name__ == "__main__":
    t, KI, KII = np.loadtxt(sys.argv[1], unpack=True)
    crack = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
    print(f"K_I {KI.min():.2f}..{KI.max():.2f}, K_II {KII.min():.2f}..{KII.max():.2f} MPa sqrt(m)")
    for k, v in all_criteria(t, KI, KII).items():
        loc = ", ".join(f"{np.degrees(x):6.1f}" for x in v)
        glob = ", ".join(f"{crack+np.degrees(x):6.1f}" for x in v)
        print(f"{k:16s} theta_c = {loc:16s} theta_g = {glob}")
