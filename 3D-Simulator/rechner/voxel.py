"""Voxel-Raster eines Modells – Gegenstück zu rasterize/markPrism in js/hornpath.js und findMouth in js/dae.js.

* Wände: Dreiecke werden dicht abgetastet (Linien parallel zur längsten Kante, Abstand ≤ 0,4 Zellen und ≤ 5 mm);
  jede getroffene Zelle ist Wand. Innenräume dicker Platten bleiben Luft, sind aber vom Rest abgeschlossen.
* Abdichtungen: achsparallele Quader/Prismen (Zellen mit Mittelpunkt im Querschnitt, entlang der Achse jede
  berührte Zelle), z. B. für Kabeldurchführungen oder zu kleine Schallwände.
* Lufträume: 6er-Nachbarschaft (Flutung um einen Punkt = Zusammenhangskomponente).
* Rasterkorrektur: eine Wandfläche belegt im Mittel eine halbe Zelle Luft. Je Wand-Nachbar (6er) bekommt eine
  Luftzelle 0,5 Zellvolumen dazu; quer dazu wird die Leitfähigkeit entsprechend erhöht (siehe fem.py).
"""
import numpy as np
from scipy import ndimage

LUFT, WAND, DICHT = 0, 1, 2
N6 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])


class Raster:
    def __init__(self, modell, h, abtastung=None, masken=None):
        """modell: dae.Modell; h: Zellgröße in m; masken: {name: Textmuster der Bauteile} für Sondermasken
        (Standard: 'membran' und 'magnet')."""
        self.modell = modell
        self.h = float(h)
        self.dims = tuple(int(max(1, np.ceil(s / h - 1e-9))) for s in modell.size)
        self.lab = np.zeros(self.dims, dtype=np.uint8)
        self.teil = np.full(self.dims, -1, dtype=np.int16)   # letztes Bauteil je Wandzelle
        if masken is None:
            masken = {'membran': ['membran'], 'magnet': ['magnet']}
        self.masken = {k: np.zeros(self.dims, dtype=bool) for k in masken}
        step = abtastung or min(0.4 * self.h, 0.005)
        self._rasterize(step, {k: modell.teil_maske(v) for k, v in masken.items()})
        self._regionen = None

    # ---------- Wände ----------
    def _rasterize(self, step, tri_masken):
        t = self.modell.tri
        if not len(t):
            return
        L = np.linalg.norm(t[:, [1, 2, 0]] - t[:, [0, 1, 2]], axis=2)
        e = np.argmax(L, axis=1)
        ar = np.arange(len(t))
        p0, p1, p2 = t[ar, e], t[ar, (e + 1) % 3], t[ar, (e + 2) % 3]
        base = np.maximum(L[ar, e], 1e-12)
        height = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1) / base
        ns = np.maximum(1, np.ceil(height / step)).astype(np.int64)
        # in Blöcken, damit der Speicher klein bleibt
        npts_est = (ns + 1) * (np.ceil(base / step) + 2)
        order = np.arange(len(t))
        lim, start = 4e6, 0
        csum = np.cumsum(npts_est)
        while start < len(t):
            stop = int(np.searchsorted(csum, (csum[start - 1] if start else 0) + lim, side='right'))
            stop = max(stop, start + 1)
            sel = order[start:stop]
            self._raster_block(sel, p0[sel], p1[sel], p2[sel], base[sel], ns[sel], step, tri_masken)
            start = stop

    def _raster_block(self, sel, p0, p1, p2, base, ns, step, tri_masken):
        nlines = ns + 1
        li = np.repeat(np.arange(len(sel)), nlines)
        a = np.arange(len(li)) - np.repeat(np.cumsum(nlines) - nlines, nlines)
        s = (a / ns[li])[:, None]
        q = p0[li] + s * (p2[li] - p0[li])
        w = p1[li] + s * (p2[li] - p1[li]) - q
        nl = np.maximum(1, np.ceil(base[li] * (1 - s[:, 0]) / step)).astype(np.int64)
        pi = np.repeat(np.arange(len(li)), nl + 1)
        b = np.arange(len(pi)) - np.repeat(np.cumsum(nl + 1) - (nl + 1), nl + 1)
        pts = q[pi] + w[pi] * (b / nl[pi])[:, None]
        ix = np.clip(np.floor(pts / self.h).astype(np.int64), 0, np.array(self.dims) - 1)
        lin = np.ravel_multi_index(ix.T, self.dims)
        tri = sel[li[pi]]
        self.lab.flat[lin] = WAND
        self.teil.flat[lin] = self.modell.part_of[tri]
        for k, m in tri_masken.items():
            if m.any():
                hit = m[tri]
                self.masken[k].flat[lin[hit]] = True

    # ---------- Abdichtungen ----------
    def abdichtung(self, achse, von, bis, rechteck=None, polygon=None, aussparung=None, wert=DICHT, mit_maske=False):
        """Quader (rechteck=[u0,u1,v0,v1]) oder Prisma (polygon=[[u,v],…]) entlang einer Achse dicht setzen.
        u/v sind die Achsen (achse+1)%3 und (achse+2)%3. aussparung: Rechteck oder Polygon (u/v), das frei
        bleibt, z. B. eine Schallwand samt Treiberloch, die bis zum Rand verlängert wird.
        Gibt die Zahl der geänderten Luftzellen zurück, mit mit_maske=True zusätzlich die Maske des ganzen Quaders."""
        a, u, v = achse, (achse + 1) % 3, (achse + 2) % 3
        h, d = self.h, self.dims
        if isinstance(polygon, str):
            polygon = self.teil_umriss(polygon, achse)
        if isinstance(aussparung, str):
            aussparung = self.teil_umriss(aussparung, achse)
        lo, hi = sorted((von, bis))
        i0 = int(np.clip(np.floor(lo / h), 0, d[a] - 1))
        i1 = int(np.clip(np.ceil(hi / h) - 1, i0, d[a] - 1))
        cu = (np.arange(d[u]) + 0.5) * h
        cv = (np.arange(d[v]) + 0.5) * h
        if polygon is not None:
            poly = np.asarray(polygon, float)
            UU, VV = np.meshgrid(cu, cv, indexing='ij')
            inside = _in_poly(UU, VV, poly)
            if not inside.any():
                c = poly.mean(axis=0)
                inside[int(np.clip(c[0] // h, 0, d[u] - 1)), int(np.clip(c[1] // h, 0, d[v] - 1))] = True
        else:
            u0, u1, v0, v1 = rechteck
            iu = _range_cells(cu, h, min(u0, u1), max(u0, u1), d[u])
            iv = _range_cells(cv, h, min(v0, v1), max(v0, v1), d[v])
            inside = np.zeros((d[u], d[v]), dtype=bool)
            inside[np.ix_(iu, iv)] = True
        if aussparung is not None:
            UU, VV = np.meshgrid(cu, cv, indexing='ij')
            ap = np.asarray(aussparung, float)
            if ap.ndim == 1:
                inside &= ~((UU >= min(ap[0], ap[1])) & (UU <= max(ap[0], ap[1]))
                            & (VV >= min(ap[2], ap[3])) & (VV <= max(ap[2], ap[3])))
            else:
                inside &= ~_in_poly(UU, VV, ap)
        sl = [slice(None)] * 3
        sl[a] = slice(i0, i1 + 1)
        # Querschnitt in die Reihenfolge (x, y, z) bringen
        mask2 = np.zeros(self.dims, dtype=bool)
        block = np.broadcast_to(inside[None], (i1 - i0 + 1,) + inside.shape)   # (a, u, v)
        perm = np.argsort([a, u, v])
        mask2[tuple(sl)] = np.transpose(block, perm)
        sub = mask2 & (self.lab == LUFT)
        self.lab[sub] = wert
        self._regionen = None
        return (int(sub.sum()), mask2) if mit_maske else int(sub.sum())

    def teil_umriss(self, muster, achse):
        """Konvexe Hülle eines Bauteils (Name enthält `muster`) in der Ebene (u, v) senkrecht zur Achse."""
        from scipy.spatial import ConvexHull
        sel = self.modell.teil_maske(muster)
        if not sel.any():
            raise ValueError('Part "%s" not found in the model' % muster)
        p = self.modell.tri[sel].reshape(-1, 3)[:, [(achse + 1) % 3, (achse + 2) % 3]]
        return p[ConvexHull(p).vertices]

    # ---------- Lufträume ----------
    def regionen(self):
        """Zusammenhängende Lufträume (6er-Nachbarschaft): (Labelfeld, Anzahl)."""
        if self._regionen is None:
            self._regionen = ndimage.label(self.lab == LUFT, structure=ndimage.generate_binary_structure(3, 1))
        return self._regionen

    def zelle(self, punkt, suche=2):
        """Zellindex eines Punkts (m); liegt er in einer Wand, die nächste Luftzelle im Umkreis von `suche` Zellen."""
        c = np.clip(np.floor(np.asarray(punkt, float) / self.h).astype(int), 0, np.array(self.dims) - 1)
        if self.lab[tuple(c)] == LUFT:
            return tuple(c)
        best, bd = None, np.inf
        for d in np.ndindex(*(2 * suche + 1,) * 3):
            q = c + np.array(d) - suche
            if np.any(q < 0) or np.any(q >= self.dims):
                continue
            if self.lab[tuple(q)] == LUFT:
                dist = np.sum((np.array(d) - suche) ** 2)
                if dist < bd:
                    best, bd = tuple(q), dist
        if best is None:
            raise ValueError('The point %s lies inside a wall (no air within %d cells)'
                             % (fmt_punkt(punkt), suche))
        return best

    def region_um(self, punkt):
        """Maske des Luftraums, in dem der Punkt liegt (Flutung)."""
        lbl, _ = self.regionen()
        return lbl == lbl[self.zelle(punkt)]

    def region_id(self, punkt):
        lbl, _ = self.regionen()
        return int(lbl[self.zelle(punkt)])

    def wandnachbarn(self):
        """Zahl der Wand-Nachbarn (nur WAND, nicht Abdichtung) je Zelle und Achse: (3, nx, ny, nz) uint8."""
        w = self.lab == WAND
        out = np.zeros((3,) + self.dims, dtype=np.uint8)
        for a in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[a], hi[a] = slice(0, -1), slice(1, None)
            out[a][tuple(lo)] += w[tuple(hi)]
            out[a][tuple(hi)] += w[tuple(lo)]
        return out

    def volumen(self, maske):
        """Luftvolumen einer Zellmenge mit Rasterkorrektur (m³)."""
        nw = self.wandnachbarn()
        return float((maske.sum() + 0.5 * (nw.sum(axis=0) * maske).sum()) * self.h ** 3)

    def offen(self, maske):
        """Berührt der Luftraum den Rand des Rasters (= Öffnung nach außen)?"""
        return bool(maske[0].any() or maske[-1].any() or maske[:, 0].any() or maske[:, -1].any()
                    or maske[:, :, 0].any() or maske[:, :, -1].any())

    def ausdehnung(self, maske):
        idx = np.argwhere(maske)
        return idx.min(axis=0) * self.h, (idx.max(axis=0) + 1) * self.h

    def teile_bei(self, zelle, radius=2):
        """Namen der Bauteile, deren Wandzellen nahe einer Zelle liegen."""
        c = np.array(zelle)
        lo = np.maximum(c - radius, 0)
        hi = np.minimum(c + radius + 1, self.dims)
        ids = np.unique(self.teil[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        return [self.modell.parts[i] for i in ids if i >= 0]

    # ---------- Leck-Suche ----------
    def leckweg(self, punkt_a, punkt_b, max_schritte=5000):
        """Kürzester Luftweg (6er) zwischen zwei Punkten im selben Luftraum und seine engsten Stellen.
        Gibt [{'punkt': [x,y,z] m, 'weite': m, 'teile': [...]}, …] zurück (engste zuerst) oder None."""
        a, b = self.zelle(punkt_a), self.zelle(punkt_b)
        lbl, _ = self.regionen()
        if lbl[a] != lbl[b]:
            return None
        reg = lbl == lbl[a]
        dist = np.full(self.dims, -1, dtype=np.int32)
        dist[a] = 0
        front = np.zeros(self.dims, dtype=bool)
        front[a] = True
        st = ndimage.generate_binary_structure(3, 1)
        k = 0
        while dist[b] < 0 and k < max_schritte:
            k += 1
            front = ndimage.binary_dilation(front, st) & reg & (dist < 0)
            if not front.any():
                return None
            dist[front] = k
        # zurückverfolgen
        path = [np.array(b)]
        cur = np.array(b)
        while dist[tuple(cur)] > 0:
            for d in N6:
                q = cur + d
                if np.all(q >= 0) and np.all(q < self.dims) and dist[tuple(q)] == dist[tuple(cur)] - 1:
                    cur = q
                    break
            path.append(cur.copy())
        path = np.array(path)
        edt = ndimage.distance_transform_edt(self.lab == LUFT) * self.h
        w = edt[tuple(path.T)]
        out, used = [], []
        for i in np.argsort(w, kind='stable'):
            if any(np.abs(path[i] - path[j]).max() <= 3 for j in used):
                continue
            used.append(i)
            out.append({'punkt': ((path[i] + 0.5) * self.h).round(4).tolist(), 'weite': float(2 * w[i]),
                        'teile': self.teile_bei(path[i])})
            if len(out) >= 3:
                break
        return {'laenge': float(len(path) * self.h), 'engstellen': out}


def _range_cells(centers, h, lo, hi, n):
    idx = np.nonzero((centers >= lo) & (centers <= hi))[0]
    if not len(idx):
        idx = np.array([int(np.clip((lo + hi) / 2 // h, 0, n - 1))])
    return idx


def _in_poly(x, y, poly):
    inside = np.zeros(x.shape, dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-300) + xi)
        inside ^= cond
        j = i
    return inside


def fmt_punkt(p):
    return '(' + ', '.join('%.1f' % (100 * v) for v in p) + ') cm'


# ---------- Mundöffnung finden (wie findMouth in js/dae.js) ----------
SEITEN = [
    {'name': 'Front (−Y)', 'a': 1, 'side': 0}, {'name': 'Back (+Y)', 'a': 1, 'side': 1},
    {'name': 'Left (−X)', 'a': 0, 'side': 0}, {'name': 'Right (+X)', 'a': 0, 'side': 1},
    {'name': 'Bottom (−Z)', 'a': 2, 'side': 0}, {'name': 'Top (+Z)', 'a': 2, 'side': 1},
]


def _depth_map(modell, face, res):
    a, side = face['a'], face['side']
    u, v = (a + 1) % 3, (a + 2) % 3
    size = modell.size
    h = max(size[u], size[v]) / res
    nu, nv = max(1, int(np.ceil(size[u] / h))), max(1, int(np.ceil(size[v] / h)))
    depth = np.full(nu * nv, np.inf)
    t = modell.tri
    U, V, A = t[:, :, u], t[:, :, v], t[:, :, a]
    det = (V[:, 1] - V[:, 2]) * (U[:, 0] - U[:, 2]) + (U[:, 2] - U[:, 1]) * (V[:, 0] - V[:, 2])
    ok = np.abs(det) >= 1e-10
    iu0 = np.maximum(0, np.floor(U.min(1) / h)).astype(np.int64)
    iu1 = np.minimum(nu - 1, np.floor(U.max(1) / h)).astype(np.int64)
    iv0 = np.maximum(0, np.floor(V.min(1) / h)).astype(np.int64)
    iv1 = np.minimum(nv - 1, np.floor(V.max(1) / h)).astype(np.int64)
    cu, cv = np.maximum(0, iu1 - iu0 + 1), np.maximum(0, iv1 - iv0 + 1)
    cnt = np.where(ok, cu * cv, 0)
    ti = np.repeat(np.arange(len(t)), cnt)
    k = np.arange(len(ti)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    iu = iu0[ti] + k // cv[ti]
    iv = iv0[ti] + k % cv[ti]
    pu, pv = (iu + 0.5) * h, (iv + 0.5) * h
    d_ = det[ti]
    l0 = ((V[ti, 1] - V[ti, 2]) * (pu - U[ti, 2]) + (U[ti, 2] - U[ti, 1]) * (pv - V[ti, 2])) / d_
    l1 = ((V[ti, 2] - V[ti, 0]) * (pu - U[ti, 2]) + (U[ti, 0] - U[ti, 2]) * (pv - V[ti, 2])) / d_
    l2 = 1 - l0 - l1
    hit = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
    pa = l0 * A[ti, 0] + l1 * A[ti, 1] + l2 * A[ti, 2]
    dd = pa if side == 0 else size[a] - pa
    np.minimum.at(depth, (iu * nv + iv)[hit], dd[hit])
    return {'depth': depth.reshape(nu, nv), 'nu': nu, 'nv': nv, 'h': h, 'u': u, 'v': v}


def find_mouth(modell, res=100):
    """Mund-Kandidaten je Außenseite der Bounding Box, nach Fläche sortiert (größter zuerst)."""
    out = []
    for fi, face in enumerate(SEITEN):
        dm = _depth_map(modell, face, res)
        ext = modell.size[face['a']]
        thr = max(0.05, 0.1 * ext)
        open_ = dm['depth'] > thr
        lbl, n = ndimage.label(open_, structure=ndimage.generate_binary_structure(2, 1))
        if not n:
            continue
        sizes = ndimage.sum(open_, lbl, index=np.arange(1, n + 1))
        if sizes.max() < 4:
            continue
        big = np.nonzero(sizes >= 0.2 * sizes.max())[0] + 1
        cells = np.isin(lbl, big)
        iu, iv = np.nonzero(cells)
        dvals = dm['depth'][cells]
        through = np.isinf(dvals).sum()
        hh = dm['h']
        out.append({
            'face': fi, 'name': face['name'], 'a': face['a'], 'side': face['side'], 'u': dm['u'], 'v': dm['v'],
            'flaeche': float(cells.sum() * hh * hh),
            'rect': {'u0': iu.min() * hh, 'u1': (iu.max() + 1) * hh, 'v0': iv.min() * hh, 'v1': (iv.max() + 1) * hh},
            'tiefe': float(dvals[np.isfinite(dvals)].mean()) if through < len(dvals) else None,
            'durchgehend': float(through / len(dvals)), 'teile': int(len(big)),
        })
    out.sort(key=lambda c: -c['flaeche'])
    return out
