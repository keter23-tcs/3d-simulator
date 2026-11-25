"""Abstrahlung über das Rayleigh-Integral (ebene Öffnung in unendlicher Schallwand = Halbraum).

Die Mundfläche (alle Zellflächen eines Luftraums auf einer Außenseite des Rasters) wird in Felder ("Patches")
mit jeweils gleichförmiger Normalschnelle v_P zerlegt. Mit dem Kern e^{-jkR}/(2πR) gilt für die über ein Feld
integrierten Drücke  F_P = ∫_P p dS = jωρ Σ_Q G_PQ v_Q,
    G_PQ = 1/(2π) ∫_P ∫_Q e^{-jkR}/R dS dS'      (symmetrisch, m³).
Für einen Kolben in der Schallwand ist das exakt die Strahlungsimpedanz wie in js/hornsim.js (R1/X1).

Statischer Anteil 1/R: Nahfeld exakt (geschlossene Formel für zwei achsparallele Rechtecke in einer Ebene),
Fernfeld per Mittelpunktregel auf Teilzellen. Dynamischer Anteil (e^{-jkR}-1)/R ist glatt und wird auf
Teilzellen integriert. Mehrere Münder in derselben Ebene (Stack) teilen sich eine Ebene und koppeln über G.
"""
import numpy as np
import scipy.sparse as sp

RHO = 1.205
C = 343.0


def _F(x, y):
    """Vierfache Stammfunktion von 1/r (∂²/∂x² ∂²/∂y² F = 1/√(x²+y²)), lineare Anteile weggelassen."""
    ax, ay = np.abs(x), np.abs(y)
    r = np.sqrt(x * x + y * y)
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = np.where(ax > 0, 0.5 * x * x * y * np.arcsinh(y / np.where(ax > 0, ax, 1)), 0.0)
        t2 = np.where(ay > 0, 0.5 * x * y * y * np.arcsinh(x / np.where(ay > 0, ay, 1)), 0.0)
    return t1 + t2 - r ** 3 / 6


def rechteck_integral(a, b):
    """∫∫ 1/|x-y| über zwei achsparallele Rechtecke a, b = (..., 4) [x0, x1, y0, y1] in derselben Ebene."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = 0.0
    for (ia, ib, sx) in ((1, 0, 1), (0, 1, 1), (0, 0, -1), (1, 1, -1)):
        dx = a[..., ia] - b[..., ib]
        for (ja, jb, sy) in ((3, 2, 1), (2, 3, 1), (2, 2, -1), (3, 3, -1)):
            dy = a[..., ja] - b[..., jb]
            out = out + sx * sy * _F(dx, dy)
    return out


# Mittlerer Abstand zweier Punkte im Einheitsquadrat (für den dynamischen Eigenanteil)
_MEAN_R_SQUARE = 0.5214054332


class Strahler:
    """Zellflächen eines Mundes in einer Ebene: Mittelpunkte (n, 2) in m (Ebenenkoordinaten u, v), Zellgröße h."""

    def __init__(self, uv, h, name=''):
        self.uv = np.asarray(uv, float).reshape(-1, 2)
        self.h = float(h)
        self.name = name

    @property
    def flaeche(self):
        return len(self.uv) * self.h * self.h


class Ebene:
    """Alle Münder einer Ebene mit gemeinsamer Rayleigh-Kopplung.

    patch: Kantenlänge der Felder gleicher Schnelle (m); teilzelle: Kantenlänge der Quadraturzellen (m)."""

    def __init__(self, strahler, patch, teilzelle=None, nah=3.0):
        self.strahler = list(strahler)
        h = self.strahler[0].h
        self.h = h
        m = max(1, int(round(patch / h)))
        nf = sum(len(s.uv) for s in self.strahler)
        if teilzelle is None:
            mq = max(1, int(np.ceil(np.sqrt(nf / 2500.0))))
        else:
            mq = max(1, int(round(teilzelle / h)))
        mq = min(mq, m)
        while m % mq:
            mq -= 1
        self.m, self.mq = m, mq
        uv, pid, sid, owner = [], [], [], []
        np_, ns_ = 0, 0
        self.patch_von = []   # (strahler-Index, erster Patch, Anzahl)
        for si, s in enumerate(self.strahler):
            g = np.floor(s.uv / h + 1e-6).astype(np.int64)      # Rasterindex der Fläche
            gp = np.floor_divide(g, m)
            gs = np.floor_divide(g, mq)
            _, pinv = np.unique(gp, axis=0, return_inverse=True)
            _, sinv = np.unique(gs, axis=0, return_inverse=True)
            pinv, sinv = pinv.ravel(), sinv.ravel()
            self.patch_von.append((si, np_, int(pinv.max()) + 1))
            uv.append(s.uv)
            pid.append(pinv + np_)
            sid.append(sinv + ns_)
            owner.append(np.full(len(s.uv), si))
            np_ += int(pinv.max()) + 1
            ns_ += int(sinv.max()) + 1
        self.uv = np.vstack(uv)
        self.patch = np.concatenate(pid)          # Patch je Fläche
        self.sub = np.concatenate(sid)            # Teilzelle je Fläche
        self.nfl, self.npatch, self.nsub = len(self.uv), np_, ns_
        a = h * h
        self.patch_flaeche = np.bincount(self.patch, minlength=np_) * a
        cnt = np.bincount(self.sub, minlength=ns_)
        self.sub_flaeche = cnt * a
        self.sub_mitte = np.stack([np.bincount(self.sub, self.uv[:, k], ns_) / cnt for k in range(2)], axis=1)
        self.sub_patch = np.zeros(ns_, dtype=np.int64)
        self.sub_patch[self.sub] = self.patch
        self.E = sp.csr_matrix((np.ones(ns_), (np.arange(ns_), self.sub_patch)), shape=(ns_, np_))
        self._statisch(nah)

    def _statisch(self, nah):
        """Frequenzunabhängiger Anteil (1/R) auf Teilzell- und Patch-Ebene."""
        h, ns = self.h, self.nsub
        sc = self.sub_mitte
        d = np.sqrt(((sc[:, None, :] - sc[None, :, :]) ** 2).sum(-1)) if ns <= 6000 else None
        lim = nah * self.mq * h * np.sqrt(2)
        A = self.sub_flaeche
        with np.errstate(divide='ignore'):
            G = np.where(d > 0, A[:, None] * A[None, :] / np.where(d > 0, d, 1), 0.0)
        near = d < lim
        G[near] = 0.0
        # Nahfeld exakt auf Flächenebene
        order = np.argsort(self.sub, kind='stable')
        starts = np.searchsorted(self.sub[order], np.arange(ns + 1))
        ia, ib = np.nonzero(near)
        keep = ia <= ib
        ia, ib = ia[keep], ib[keep]
        rect = np.concatenate([self.uv - h / 2, self.uv + h / 2], axis=1)[:, [0, 2, 1, 3]]
        fa_list, fb_list, pa_list, pb_list = [], [], [], []
        for a_, b_ in zip(ia, ib):
            fa = order[starts[a_]:starts[a_ + 1]]
            fb = order[starts[b_]:starts[b_ + 1]]
            FA, FB = np.meshgrid(fa, fb, indexing='ij')
            fa_list.append(FA.ravel())
            fb_list.append(FB.ravel())
            pa_list.append(np.full(FA.size, a_))
            pb_list.append(np.full(FA.size, b_))
        if fa_list:
            fa, fb = np.concatenate(fa_list), np.concatenate(fb_list)
            pa, pb = np.concatenate(pa_list), np.concatenate(pb_list)
            val = rechteck_integral(rect[fa], rect[fb])
            add = np.zeros((ns, ns))
            np.add.at(add, (pa, pb), val)
            add = add + np.triu(add, 1).T
            G += add
        self.d_sub = d
        self.G0_sub = G / (2 * np.pi)
        self.G0 = (self.E.T @ (self.E.T @ self.G0_sub).T).T

    def G(self, k):
        """Kopplungsmatrix der Patches (npatch × npatch, komplex) für die Wellenzahl k."""
        d, A = self.d_sub, self.sub_flaeche
        with np.errstate(divide='ignore', invalid='ignore'):
            dyn = np.where(d > 0, (np.exp(-1j * k * d) - 1) / np.where(d > 0, d, 1), 0)
        self_term = -1j * k - 0.5 * k * k * _MEAN_R_SQUARE * np.sqrt(A)
        dyn[np.diag_indices_from(dyn)] = self_term
        Gd = dyn * (A[:, None] * A[None, :]) / (2 * np.pi)
        Gd = (self.E.T @ (self.E.T @ Gd).T).T
        return self.G0 + Gd


def kolben_impedanz(S, k):
    """Akustische Strahlungsimpedanz eines Kreiskolbens (Fläche S) in unendlicher Schallwand (Pa·s/m³)."""
    from scipy.special import j1, struve
    a = np.sqrt(S / np.pi)
    x = 2 * np.asarray(k, float) * a
    z0 = RHO * C / S
    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = np.where(x < 1e-3, x * x / 8, 1 - 2 * j1(x) / np.where(x > 0, x, 1))
        x1 = np.where(x < 1e-3, 4 * x / (3 * np.pi), 2 * struve(1, x) / np.where(x > 0, x, 1))
    return z0 * (r1 + 1j * x1)
