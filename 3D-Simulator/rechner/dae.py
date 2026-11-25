"""Import von 3D-Modellen (.dae / COLLADA, z. B. aus SketchUp) – Gegenstück zu RS.dae.parse in js/dae.js.

Liefert Dreiecke in Metern (Z nach oben, Ursprung = Ecke der Bounding Box) und je Dreieck den Namen des
nächsten benannten Knotens (Bauteil), z. B. "Treiber links – Membran".
Polygone (polylist/polygons, auch mit Löchern <ph>/<h>) werden per Ohrenschneiden trianguliert.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Modell:
    tri: np.ndarray            # (n, 3, 3) Eckpunkte in m
    size: np.ndarray           # Ausdehnung [x, y, z] in m
    einheit: float             # Meter je Modelleinheit
    up: str                    # Hochachse der Datei
    parts: list = field(default_factory=list)   # Bauteilnamen
    part_of: np.ndarray = None  # (n,) Index in parts je Dreieck

    def teil_maske(self, muster):
        """Dreiecke, deren Bauteilname einen der Texte (ohne Groß-/Kleinschreibung) enthält."""
        if isinstance(muster, str):
            muster = [muster]
        treffer = [i for i, n in enumerate(self.parts) if any(m.lower() in (n or '').lower() for m in muster)]
        return np.isin(self.part_of, treffer)


def _local(tag):
    return tag.rsplit('}', 1)[-1]


def _kids(el, tag):
    return [c for c in el if _local(c.tag) == tag]


def _kid(el, tag):
    k = _kids(el, tag)
    return k[0] if k else None


def _nums(text):
    return np.array(text.split(), dtype=float) if text and text.strip() else np.zeros(0)


def _rotation(ax, ay, az, deg):
    l = np.sqrt(ax * ax + ay * ay + az * az) or 1.0
    x, y, z = ax / l, ay / l, az / l
    a = np.radians(deg)
    c, s, t = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0],
                     [0, 0, 0, 1.0]])


def _node_matrix(node):
    m = np.eye(4)
    for c in node:
        tag = _local(c.tag)
        if tag == 'matrix':
            m = m @ _nums(c.text).reshape(4, 4)
        elif tag == 'translate':
            v = _nums(c.text)
            t = np.eye(4)
            t[:3, 3] = v[:3]
            m = m @ t
        elif tag == 'scale':
            v = _nums(c.text)
            m = m @ np.diag([v[0], v[1], v[2], 1.0])
        elif tag == 'rotate':
            v = _nums(c.text)
            m = m @ _rotation(*v[:4])
    return m


# ---------- Triangulierung von Polygonen (Ohrenschneiden, Löcher über Brücken) ----------

def _plane_2d(pts):
    """Projektion auf die Ebene mit größter Fläche (Newell-Normale)."""
    n = np.zeros(3)
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
    k = int(np.argmax(np.abs(n)))
    u, v = (k + 1) % 3, (k + 2) % 3
    return pts[:, [u, v]], n[k]


def _area2(p):
    x, y = p[:, 0], p[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _in_tri(p, a, b, c):
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    neg = d1 < -1e-14 or d2 < -1e-14 or d3 < -1e-14
    pos = d1 > 1e-14 or d2 > 1e-14 or d3 > 1e-14
    return not (neg and pos)


def _earclip(p2, idx):
    """p2: 2D-Punkte (gegen Uhrzeigersinn), idx: zugehörige Indizes → Liste von Index-Dreiecken."""
    idx = list(idx)
    out = []
    guard = 0
    while len(idx) > 3 and guard < 10 * len(p2) + 100:
        guard += 1
        n = len(idx)
        found = False
        for i in range(n):
            ia, ib, ic = idx[i - 1], idx[i], idx[(i + 1) % n]
            a, b, c = p2[ia], p2[ib], p2[ic]
            cr = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cr <= 1e-14:
                continue
            if any(_in_tri(p2[j], a, b, c) for j in idx if j not in (ia, ib, ic)
                   and not (np.allclose(p2[j], a) or np.allclose(p2[j], b) or np.allclose(p2[j], c))):
                continue
            out.append((ia, ib, ic))
            del idx[i]
            found = True
            break
        if not found:
            break
    if len(idx) == 3:
        out.append(tuple(idx))
    elif len(idx) > 3:  # entartet: Rest als Fächer
        for k in range(1, len(idx) - 1):
            out.append((idx[0], idx[k], idx[k + 1]))
    return out


def triangulate(outer, holes=()):
    """outer: (n,3)-Punkte, holes: Liste von (m,3)-Punkten → (k,3,3) Dreiecke."""
    outer = np.asarray(outer, float)
    if len(outer) < 3:
        return np.zeros((0, 3, 3))
    if len(outer) == 3 and not holes:
        return outer[None]
    allp = [outer] + [np.asarray(h, float) for h in holes if len(h) >= 3]
    pts = np.vstack(allp)
    p2, _ = _plane_2d(pts)
    starts = np.cumsum([0] + [len(a) for a in allp])
    rings = [list(range(starts[i], starts[i + 1])) for i in range(len(allp))]
    if _area2(p2[rings[0]]) < 0:
        rings[0] = rings[0][::-1]
    poly = rings[0]
    # Löcher (im Uhrzeigersinn) über eine Brücke vom rechtesten Lochpunkt einhängen
    holes_r = []
    for r in rings[1:]:
        if _area2(p2[r]) > 0:
            r = r[::-1]
        holes_r.append(r)
    holes_r.sort(key=lambda r: -max(p2[j][0] for j in r))
    for r in holes_r:
        k = max(range(len(r)), key=lambda j: p2[r[j]][0])
        hp = p2[r[k]]
        # sichtbarer Randpunkt: nächster Punkt rechts vom Lochpunkt, dessen Verbindung keine Kante schneidet
        best, bd = None, np.inf
        for pi, j in enumerate(poly):
            d = p2[j] - hp
            if d[0] < -1e-12:
                continue
            dist = d @ d
            if dist < bd and not _crosses(p2, poly, hp, p2[j]):
                best, bd = pi, dist
        if best is None:
            best = int(np.argmin([np.sum((p2[j] - hp) ** 2) for j in poly]))
        hr = r[k:] + r[:k]
        poly = poly[:best + 1] + hr + [hr[0], poly[best]] + poly[best + 1:]
    tris = _earclip(p2, poly)
    return np.array([[pts[a], pts[b], pts[c]] for a, b, c in tris]).reshape(-1, 3, 3)


def _crosses(p2, poly, a, b):
    n = len(poly)
    for i in range(n):
        c, d = p2[poly[i]], p2[poly[(i + 1) % n]]
        if np.allclose(c, b) or np.allclose(d, b) or np.allclose(c, a) or np.allclose(d, a):
            continue
        o1 = np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        o2 = np.sign((b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]))
        o3 = np.sign((d[0] - c[0]) * (a[1] - c[1]) - (d[1] - c[1]) * (a[0] - c[0]))
        o4 = np.sign((d[0] - c[0]) * (b[1] - c[1]) - (d[1] - c[1]) * (b[0] - c[0]))
        if o1 != o2 and o3 != o4:
            return True
    return False


# ---------- Geometrie lesen ----------

def _read_geometry(geom):
    """Alle Dreiecke einer <geometry> in lokalen Koordinaten, (n,3,3)."""
    mesh = _kid(geom, 'mesh')
    if mesh is None:
        return np.zeros((0, 3, 3))
    sources = {}
    for s in _kids(mesh, 'source'):
        fa = _kid(s, 'float_array')
        acc = next((e for e in s.iter() if _local(e.tag) == 'accessor'), None)
        if fa is not None:
            sources[s.get('id')] = (_nums(fa.text), int(acc.get('stride', 3)) if acc is not None else 3)
    vert_src = {}
    for v in _kids(mesh, 'vertices'):
        for inp in _kids(v, 'input'):
            if inp.get('semantic') == 'POSITION':
                vert_src[v.get('id')] = inp.get('source')[1:]
    out = []
    for prim in mesh:
        typ = _local(prim.tag)
        if typ not in ('triangles', 'polylist', 'polygons', 'trifans', 'tristrips'):
            continue
        stride, voff, src = 0, 0, None
        for inp in _kids(prim, 'input'):
            off = int(inp.get('offset', 0))
            stride = max(stride, off + 1)
            if inp.get('semantic') == 'VERTEX':
                voff = off
                sid = inp.get('source')[1:]
                src = sources.get(vert_src.get(sid, sid))
        if src is None:
            continue
        data, sstride = src
        pos = data[:len(data) // sstride * sstride].reshape(-1, sstride)[:, :3]
        if typ == 'triangles':
            pe = _kid(prim, 'p')
            p = _nums(pe.text if pe is not None else '').astype(np.int64)
            ix = p[voff::stride]
            ix = ix[:len(ix) // 3 * 3]
            out.append(pos[ix].reshape(-1, 3, 3))
            continue
        polys = []  # (äußerer Ring, [Löcher]) als Indexlisten
        if typ == 'polylist':
            vc = _nums(_kid(prim, 'vcount').text).astype(int)
            pl = _nums(_kid(prim, 'p').text).astype(np.int64)[voff::stride]
            at = 0
            for cnt in vc:
                polys.append((pl[at:at + cnt], []))
                at += cnt
        elif typ == 'polygons':
            for pe in _kids(prim, 'p'):
                polys.append((_nums(pe.text).astype(np.int64)[voff::stride], []))
            for ph in _kids(prim, 'ph'):
                pe = _kid(ph, 'p')
                if pe is None:
                    continue
                holes = [_nums(h.text).astype(np.int64)[voff::stride] for h in _kids(ph, 'h')]
                polys.append((_nums(pe.text).astype(np.int64)[voff::stride], holes))
        else:  # trifans / tristrips
            for pe in _kids(prim, 'p'):
                ix = _nums(pe.text).astype(np.int64)[voff::stride]
                for k in range(1, len(ix) - 1):
                    if typ == 'trifans':
                        tri = (ix[0], ix[k], ix[k + 1])
                    else:
                        tri = (ix[k - 1], ix[k], ix[k + 1]) if k % 2 else (ix[k], ix[k - 1], ix[k + 1])
                    out.append(pos[list(tri)][None])
            continue
        for ring, holes in polys:
            if len(ring) < 3:
                continue
            if len(ring) == 3 and not holes:
                out.append(pos[ring][None])
            else:
                out.append(triangulate(pos[ring], [pos[h] for h in holes]))
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def parse(path_or_text):
    """Liest eine .dae-Datei (Pfad oder XML-Text) und gibt ein Modell zurück."""
    if isinstance(path_or_text, str) and path_or_text.lstrip().startswith('<'):
        root = ET.fromstring(path_or_text)
    else:
        root = ET.parse(path_or_text).getroot()
    unit_el = next((e for e in root.iter() if _local(e.tag) == 'unit'), None)
    up_el = next((e for e in root.iter() if _local(e.tag) == 'up_axis'), None)
    unit = float(unit_el.get('meter', 1)) if unit_el is not None else 1.0
    unit = unit or 1.0
    up = up_el.text.strip() if up_el is not None and up_el.text else 'Y_UP'

    geoms = {g.get('id'): _read_geometry(g) for g in root.iter() if _local(g.tag) == 'geometry'}
    nodes = {n.get('id'): n for n in root.iter() if _local(n.tag) == 'node' and n.get('id')}

    tris, part_of, parts = [], [], []
    part_ix = {}

    def pindex(name):
        if name not in part_ix:
            part_ix[name] = len(parts)
            parts.append(name)
        return part_ix[name]

    def walk(node, m, depth, name):
        if depth > 64:
            return
        m = m @ _node_matrix(node)
        name = node.get('name') or name
        for c in node:
            tag = _local(c.tag)
            if tag == 'instance_geometry':
                t = geoms.get((c.get('url') or '')[1:])
                if t is None or not len(t):
                    continue
                pts = t.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
                tris.append(pts.reshape(-1, 3, 3))
                part_of.append(np.full(len(t), pindex(name or ''), dtype=np.int32))
            elif tag == 'instance_node':
                ref = nodes.get((c.get('url') or '')[1:])
                if ref is not None:
                    walk(ref, m, depth + 1, name)
            elif tag == 'node':
                walk(c, m, depth + 1, name)

    scene = None
    inst = next((e for e in root.iter() if _local(e.tag) == 'instance_visual_scene'), None)
    if inst is not None:
        sid = (inst.get('url') or '')[1:]
        scene = next((e for e in root.iter() if _local(e.tag) == 'visual_scene' and e.get('id') == sid), None)
    if scene is None:
        scene = next((e for e in root.iter() if _local(e.tag) == 'visual_scene'), None)
    if scene is None:
        raise ValueError('No scene found in the .dae file')
    for n in _kids(scene, 'node'):
        walk(n, np.eye(4), 0, None)
    if not tris:
        raise ValueError('The model contains no faces')

    tri = np.concatenate(tris) * unit
    x, y, z = tri[..., 0].copy(), tri[..., 1].copy(), tri[..., 2].copy()
    if up == 'Y_UP':
        tri = np.stack([x, -z, y], axis=-1)
    elif up == 'X_UP':
        tri = np.stack([y, z, x], axis=-1)
    # entartete Dreiecke (Fläche 0) entfernen
    part_of = np.concatenate(part_of)
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ok = np.linalg.norm(cr, axis=1) > 1e-14
    tri, part_of = tri[ok], part_of[ok]
    lo = tri.reshape(-1, 3).min(axis=0)
    tri = tri - lo
    size = tri.reshape(-1, 3).max(axis=0)
    return Modell(tri=tri, size=size, einheit=unit, up=up, parts=parts, part_of=part_of)


def aus_dreiecken(tri, part_of=None, parts=None, verschieben=True):
    """Modell aus eigenen Dreiecken (m, Z oben) bauen, z. B. für Tests."""
    tri = np.asarray(tri, float).reshape(-1, 3, 3)
    if verschieben:
        tri = tri - tri.reshape(-1, 3).min(axis=0)
    size = tri.reshape(-1, 3).max(axis=0)
    if part_of is None:
        part_of = np.zeros(len(tri), dtype=np.int32)
        parts = parts or ['']
    return Modell(tri=tri, size=size, einheit=1.0, up='Z_UP', parts=list(parts), part_of=np.asarray(part_of, np.int32))
