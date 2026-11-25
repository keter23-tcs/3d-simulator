"""Treiber (Thiele/Small) wie js/driver.js und Kopplung von N Treibern an die akustische Last aus der FEM.

Je Treiber i (Schnelle v_i > 0 = Membran bewegt sich in die Druckkammer, Volumenfluss U_i = Sd·v_i):
    mechanisch:  Bl·i_i = (Rms + jωMmd + 1/(jωCms))·v_i + Sd_i · Σ_j Z_ij·Sd_j·v_j
    elektrisch:  u_i = Ze·i_i + Bl·v_i,   Ze = Re + jωLe
Z_ij = mittlerer Druckunterschied (Druckkammer − Rückkammer) an Membran i je Einheits-Volumenfluss von j.
Verschaltung parallel: u_i = π_i·V; seriell: gleicher Strom, Σ π_i·u_i = V (π = Polarität ±1).
Masse Mmd ohne Luftlast, weil die FEM die Luftlast auf beiden Seiten selbst rechnet.
"""
import numpy as np

from strahlung import RHO, C


def has(v):
    return isinstance(v, (int, float)) and np.isfinite(v) and v > 0


def si(d):
    """Datenblattwerte → SI, fehlende Größen ableiten (wie RS.driver.si)."""
    fs = d['fs']
    ws = 2 * np.pi * fs
    rc2 = RHO * C * C
    sd = d['sd'] * 1e-4 if has(d.get('sd')) else None
    mms = d['mms'] * 1e-3 if has(d.get('mms')) else None
    cms = d['cms'] * 1e-3 if has(d.get('cms')) else None
    if not cms and has(d.get('vas')) and sd:
        cms = d['vas'] * 1e-3 / (rc2 * sd * sd)
    if not mms and cms:
        mms = 1 / (ws * ws * cms)
    if not cms and mms:
        cms = 1 / (ws * ws * mms)
    bl = d['bl'] if has(d.get('bl')) else (np.sqrt(ws * mms * d['re'] / d['qes']) if mms and has(d.get('qes')) else None)
    rms = ws * mms / d['qms'] if mms and has(d.get('qms')) else None
    if not rms and mms and has(d.get('qts')) and bl:
        qes = ws * mms * d['re'] / (bl * bl)
        qms = 1 / (1 / d['qts'] - 1 / qes)
        rms = ws * mms / qms if qms > 0 else None
    out = {'re': d['re'], 'le': d['le'] * 1e-3 if has(d.get('le')) else 0.0, 'bl': bl, 'mms': mms, 'cms': cms,
           'rms': rms, 'sd': sd, 'xmax': d['xmax'] * 1e-3 if has(d.get('xmax')) else None,
           'pe': d['pe'] if has(d.get('pe')) else None, 'znom': d['znom'] if has(d.get('znom')) else d['re']}
    if mms and sd:
        a = np.sqrt(sd / np.pi)
        out['mmd'] = max(mms * 0.5, mms - 16 * RHO * a ** 3 / 3)
    return out


def fehlend(d):
    s, miss = si(d), []
    if not has(d.get('re')):
        miss.append('Re')
    if not has(d.get('fs')):
        miss.append('fs')
    if not s['sd']:
        miss.append('Sd')
    if not s['bl']:
        miss.append('Bl (or Qes + Mms)')
    if not s['mms']:
        miss.append('Mms (or Cms/Vas)')
    if not s['rms']:
        miss.append('Qms')
    return miss


def antwort(freqs, Z, H, treiber, polaritaet=None, verschaltung='parallel', volt=2.83):
    """freqs (F), Z (F, N, N) Pa·s/m³, H (F, N) Pa je m³/s in 1 m; treiber: Liste von SI-Dicts (je Einbau).
    Gibt ein Dict wie sim.ergebnis zurück (f, db, ph, x, zin, maxspl) plus Werte je Treiber."""
    N = len(treiber)
    pol = np.ones(N) if polaritaet is None else np.asarray(polaritaet, float)
    sd = np.array([t['sd'] for t in treiber])
    bl = np.array([t['bl'] for t in treiber])
    out = {'f': list(map(float, freqs)), 'db': [], 'ph': [], 'x': [], 'zin': [], 'maxspl': [],
           'x_treiber': [], 'u_treiber': []}
    for fi, f in enumerate(freqs):
        w = 2 * np.pi * f
        ze = np.array([t['re'] + 1j * w * t['le'] for t in treiber])
        zm = np.array([t['rms'] + 1j * w * (t.get('mmd') or t['mms']) + 1 / (1j * w * t['cms']) for t in treiber])
        Zmech = np.diag(zm) + sd[:, None] * Z[fi] * sd[None, :]
        if verschaltung == 'seriell':
            # Unbekannte: v (N), Strom i; Bl·π·i = Zmech v; V = Σ ze·i + Σ π·Bl·v
            A = np.zeros((N + 1, N + 1), complex)
            A[:N, :N] = Zmech
            A[:N, N] = -bl * pol
            A[N, :N] = pol * bl
            A[N, N] = ze.sum()
            rhs = np.zeros(N + 1, complex)
            rhs[N] = volt
            x = np.linalg.solve(A, rhs)
            v, cur = x[:N], x[N]
            u = ze * pol * cur + bl * v
            i_amp = cur
        else:
            # u_i = π_i V; i_i = (u_i − Bl v_i)/ze; Bl i_i = Zmech v
            A = Zmech + np.diag(bl * bl / ze)
            rhs = bl * pol * volt / ze
            v = np.linalg.solve(A, rhs)
            u = pol * volt
            i_amp = np.sum(pol * (u - bl * v) / ze)
        p = H[fi] @ (sd * v)
        spl = 20 * np.log10(max(abs(p), 1e-12) / 2e-5)
        xpk = np.abs(v) / w
        # Grenzen: Xmax je Treiber, Pe je Treiber (Spannung an den Klemmen ≤ √(Pe·Znom))
        lim = []
        for t, xi, ui in zip(treiber, xpk, u):
            if t.get('xmax') and xi > 0:
                lim.append(volt * t['xmax'] / xi)
            if t.get('pe') and abs(ui) > 0:
                lim.append(volt * np.sqrt(t['pe'] * t['znom']) / abs(ui))
        vmax = min(lim) if lim else None
        out['db'].append(spl)
        out['ph'].append(float(np.angle(p)))
        out['x'].append(float(xpk.max() * 1000))
        out['zin'].append(float(abs(volt / i_amp)))
        out['maxspl'].append(spl + 20 * np.log10(vmax / volt) if vmax else None)
        out['x_treiber'].append((xpk * 1000).tolist())
        out['u_treiber'].append(np.abs(u).tolist())
    return out
