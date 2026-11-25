"""1D-Hornsimulation – Python-Nachbau von js/hornsim.js (für Vergleiche und Tests).

Kettenmatrizen kurzer Zylinder je Segment (mit Wandverlusten), Druck-/Rückkammer als Nachgiebigkeit,
Mund als Kreiskolben in unendlicher Schallwand (Halbraum). Gleiche Formeln und Einheiten wie die App.
Bauformen 'flh', 'th', 'geschlossen', 'reflex' (Direktstrahler + Kanal) und 'hybrid' (FLH, Rückkammer mit Kanal).
"""
import numpy as np

import treiber as tr
from strahlung import RHO, C, kolben_impedanz


def log_grid(fmin, fmax, ppo):
    n = int(round(np.log2(fmax / fmin) * ppo))
    return [fmin * 2 ** (i / ppo) for i in range(n + 1)]


def _area(seg, t):
    s1, s2 = seg['s1'] * 1e-4, seg['s2'] * 1e-4
    form = seg.get('form', 'lin')
    if form == 'exp' and s1 > 0 and s2 > 0:
        return s1 * (s2 / s1) ** t
    if form == 'lin':
        return s1 + (s2 - s1) * t
    r = np.sqrt(s1) + (np.sqrt(s2) - np.sqrt(s1)) * t
    return r * r


def segment_matrix(seg, f, losses=True):
    L = seg['l'] / 100
    n = int(min(400, max(12, np.ceil(L / 0.005))))
    dl, w = L / n, 2 * np.pi * f
    M = np.eye(2, dtype=complex)
    for i in range(n):
        S = _area(seg, (i + 0.5) / n)
        z0 = RHO * C / S
        alpha = 3.0e-5 * np.sqrt(f) / np.sqrt(S / np.pi) if losses else 0.0
        kl = w / C * dl - 1j * alpha * dl
        co, s = np.cos(kl), np.sin(kl)
        M = M @ np.array([[co, 1j * s * z0], [1j * s / z0, co]])
    return M


def _zin(M, L):
    if L is None:
        return M[0, 0] / M[1, 0]
    return (M[0, 0] * L + M[0, 1]) / (M[1, 0] * L + M[1, 1])


def _rev(M):
    return np.array([[M[1, 1], M[0, 1]], [M[1, 0], M[0, 0]]])


def _pratio(M, L):
    return 1 / (M[0, 0] if L is None else M[0, 0] + M[0, 1] / L)


def _par(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a * b / (a + b)


def compliance(V, w):
    if not (V and V > 0):
        return None
    return -1j / (w * V * 1e-3 / (RHO * C * C))


def _network(segs, shunts, zmouth, f, losses):
    n = len(segs)
    Ms = [segment_matrix(s, f, losses) for s in segs]
    zdn, zup = [None] * (n + 1), [None] * (n + 1)
    zdn[n] = zmouth
    for k in range(n - 1, -1, -1):
        zdn[k] = _zin(Ms[k], _par(zdn[k + 1], shunts[k + 1]))
    for k in range(1, n + 1):
        zup[k] = _zin(_rev(Ms[k - 1]), _par(zup[k - 1], shunts[k - 1]))

    def transfer(j):
        p = [None] * (n + 1)
        p[j] = _par(_par(zup[j], zdn[j]), shunts[j])
        for q in range(j, n):
            p[q + 1] = p[q] * _pratio(Ms[q], _par(zdn[q + 1], shunts[q + 1]))
        for q in range(j, 0, -1):
            p[q - 1] = p[q] * _pratio(_rev(Ms[q - 1]), _par(zup[q - 1], shunts[q - 1]))
        return p
    return transfer


QL = 10  # Kammerverluste als Leck bei der Abstimmfrequenz, wie in der App


def abstimmung(cfg):
    """Abstimmfrequenz (Hz) aus Kammervolumen und Reflexkanal, None ohne vollständige Angaben."""
    S, L, V = (cfg.get('kanalS') or 0) * 1e-4, (cfg.get('kanalL') or 0) / 100, (cfg.get('vrc') or 0) * 1e-3
    if not (S > 0 and L > 0 and V > 0):
        return None
    return C / (2 * np.pi) * np.sqrt(S / (V * (L + 1.46 * np.sqrt(S / np.pi))))


def _reflex_kanal(cfg, stapel, losses):
    """Kanal als Rohr (innen Endkorrektur 0,61·a, außen strahlt er als Kolben ab), Leck mit Ql parallel zur Kammer."""
    fb = abstimmung(cfg)
    if not fb:
        raise ValueError('Reflex port: enter chamber volume, port area and port length')
    S = cfg['kanalS'] * 1e-4
    a = np.sqrt(S / np.pi)
    rohr = {'l': cfg['kanalL'] + 61 * a, 's1': cfg['kanalS'], 's2': cfg['kanalS'], 'form': 'lin'}
    leck = QL / (2 * np.pi * fb * (cfg['vrc'] * 1e-3 / (RHO * C * C)))

    def at(f, k):
        M = segment_matrix(rohr, f, losses)
        zrad = kolben_impedanz(S * stapel, k) * stapel
        return _zin(M, zrad), _pratio(M, zrad) / zrad
    return S, fb, leck, at


def combine(s, n, wiring='parallel'):
    if not n or n == 1:
        return s
    o = dict(s)
    o['mms'] = s['mms'] * n
    o['mmd'] = s['mmd'] * n
    o['cms'] = s['cms'] / n
    o['rms'] = s['rms'] * n
    o['sd'] = s['sd'] * n
    if wiring == 'seriell':
        o['re'], o['le'], o['bl'], o['znom'] = s['re'] * n, s['le'] * n, s['bl'] * n, s['znom'] * n
    else:
        o['re'], o['le'], o['znom'] = s['re'] / n, s['le'] / n, s['znom'] / n
    o['pe'] = s['pe'] * n if s.get('pe') else None
    return o


def simulate(cfg, driver, freqs, volt=2.83):
    """Wie RS.hornsim.simulate: cfg = profil.sim, driver = Datenblatt-Dict."""
    d = combine(tr.si(driver), cfg.get('treiberAnzahl', 1), cfg.get('verschaltung'))
    typ = cfg.get('typ', 'flh')
    direkt = typ in ('geschlossen', 'reflex')
    segs = [] if direkt else [s for s in cfg.get('segmente', []) if s['l'] > 0 and s['s1'] > 0 and s['s2'] > 0]
    if not direkt and not segs:
        raise ValueError('No valid horn segments entered')
    n = len(segs)
    stapel = max(1, cfg.get('stapel', 1))
    losses = cfg.get('verluste', True) is not False
    kanal = _reflex_kanal(cfg, stapel, losses) if typ in ('reflex', 'hybrid') else None
    mouthS = segs[-1]['s2'] * 1e-4 if n else d['sd']
    front = min(n, max(0, cfg.get('frontNode', 0))) if typ == 'th' else 0
    rear = min(n, max(0, cfg.get('rearNode', 1))) if typ == 'th' else -1
    mmd = d.get('mmd') or d['mms']
    out = {k: [] for k in ('db', 'ph', 'x', 'zin', 'maxspl')}
    out['f'] = list(freqs)
    out['kanalv'] = [] if kanal else None
    out['fb'] = kanal[1] if kanal else None
    for f in freqs:
        w = 2 * np.pi * f
        k = w / C
        zm_ = kolben_impedanz(mouthS * stapel, k) * stapel
        shunts = [None] * (n + 1)
        zr = tfk = None
        if typ == 'th':
            shunts[front] = _par(shunts[front], compliance(cfg.get('vtc'), w))
            shunts[rear] = _par(shunts[rear], compliance(cfg.get('vrc'), w))
            tf = _network(segs, shunts, zm_, f, losses)
            ta, tb = tf(front), tf(rear)
            zload = (ta[front] - tb[front]) + (tb[rear] - ta[rear])
            pm = ta[n] - tb[n]
        else:
            # Druckkammer nur vor einem Horn; beim Direktstrahler strahlt die Membran selbst ab
            shunts[0] = None if direkt else compliance(cfg.get('vtc'), w)
            t0 = _network(segs, shunts, zm_, f, losses)(0)
            zr = compliance(cfg.get('vrc'), w)
            if kanal:
                zk, tfk = kanal[3](f, k)
                zr = _par(_par(zr, kanal[2]), zk)
            zload = t0[0] + zr if zr is not None else t0[0]
            pm = t0[n]
        ze = d['re'] + 1j * w * d['le']
        zmech = d['rms'] + 1j * (w * mmd - 1 / (w * d['cms'])) + zload * d['sd'] ** 2
        v = volt * d['bl'] / (ze * zmech + d['bl'] ** 2)
        U = v * d['sd']
        um = pm * U / zm_
        if kanal:
            # Kammerdruck = −U·zr, daraus der Fluss aus dem Kanal; phasenrichtig zum Mund addiert
            uk = -U * zr * tfk
            out['kanalv'].append(abs(uk) / kanal[0])
            um = um + uk
        p = 1j * w * RHO / (2 * np.pi) * um * stapel
        spl = 20 * np.log10(max(abs(p), 1e-12) / 2e-5)
        xpk = abs(v) / w
        zin = ze + d['bl'] ** 2 / zmech
        vpe = np.sqrt(d['pe'] * d['znom']) if d.get('pe') else np.inf
        vx = volt * d['xmax'] / xpk if d.get('xmax') else np.inf
        vmax = min(vpe, vx)
        out['db'].append(spl)
        out['ph'].append(float(np.angle(p)))
        out['x'].append(xpk * 1000)
        out['zin'].append(abs(zin))
        out['maxspl'].append(spl + 20 * np.log10(vmax / volt) if np.isfinite(vmax) else None)
    return out
