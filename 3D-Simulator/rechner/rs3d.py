"""3D frequency response of a box from a SketchUp model (.dae) – command line.

    python rechner/rs3d.py auftrag.json          (with an NVIDIA GPU and CuPy the most expensive step runs there)
    python rechner/rs3d.py auftrag.json --cpu    (CPU only)

A job from the app ("Export 3D job") usually lies in the Downloads folder; then the file name is enough.
The .dae is searched next to the job, in datenblaetter/ or anywhere in the project folder.

Job (coordinates in m in the model system of js/dae.js: Z up, origin = corner of the bounding box; the JSON keys are German on purpose,
they are the format the app writes):
{
  "dae": "pfad/zum/modell.dae",
  "aufloesung": 0.01,
  "treiber": [ {T/S-Parameter wie bibliothek/treiber/*.json} ],      (one applies to all installations)
  "einbau": [ {"druckkammer": [x,y,z], "rueckkammer": [x,y,z], "polaritaet": 1}, ... ],
            (without a rear chamber, e.g. Paraflex: "rueck_offen": true instead of "rueckkammer"; needs the diaphragm)
  "abdichtungen": [ {"achse": 0|1|2, "von": m, "bis": m, "rechteck": [u0,u1,v0,v1]}, ... ],
  "verschaltung": "parallel",
  "fmin": 20, "fmax": 200, "punkte_pro_oktave": 12
}
Optional: "box" (template .box.json), "ausgabe" (target file), "name", "mund_patch" (m), "methode"
("auto" | "direkt" | "reduziert"); seals also with "polygon" [[u,v],…] or a part name and
"aussparung" (cut-out: rectangle, polygon or part name, stays free). An installation optionally with "membran" (part name
or point, if the assignment is ambiguous). Small openings to the outside (below 100 cm² and below 2 % of the
largest one, e.g. connector cut-outs) count as closed; "kleine_oeffnungen_zu": false keeps them open.
Polarity +1: a positive voltage pushes the diaphragm into the pressure chamber.

Result: <ausgabe>.box.json in the app format with profil.sim.ergebnis3d = {f, db, ph, x, zin, maxspl, ...}.
"""
import copy
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

HIER = os.path.dirname(os.path.abspath(__file__))
if HIER not in sys.path:
    sys.path.insert(0, HIER)

import dae  # noqa: E402
import fem  # noqa: E402
import hornsim1d  # noqa: E402
import strahlung  # noqa: E402
import treiber as tr  # noqa: E402
import voxel  # noqa: E402
from strahlung import C  # noqa: E402

DIRS = [(a, s) for a in range(3) for s in (1, 0)]


def _shift(mask, a, s):
    """mask shifted by one cell: out[c] = mask[c + e_a] (s=1) bzw. mask[c − e_a] (s=0)."""
    out = np.zeros_like(mask)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if s == 1:
        src[a], dst[a] = slice(1, None), slice(0, -1)
    else:
        src[a], dst[a] = slice(0, -1), slice(1, None)
    out[tuple(dst)] = mask[tuple(src)]
    return out


def flaechen_zu(region, ziel):
    """Flächen der Zellen in `region`, deren 6er-Nachbar in `ziel` liegt: (zellen, achse, seite)."""
    cells, ax, sd = [], [], []
    for a, s in DIRS:
        hit = region & _shift(ziel, a, s)
        c = np.argwhere(hit)
        cells.append(c)
        ax.append(np.full(len(c), a))
        sd.append(np.full(len(c), s))
    return np.concatenate(cells), np.concatenate(ax), np.concatenate(sd)


def _dilate(m):
    return ndimage.binary_dilation(m, ndimage.generate_binary_structure(3, 1))


# ---------- Einbau: Druck-/Rückkammer und Membran je Treiber ----------

def offene_rueckseite(raster, i, e, lbl, mem_lbl, nmem, log=print):
    """Treiber ohne Rückkammer (z. B. Paraflex): die Rückseite der Membran strahlt in den Luftraum dahinter, der auch mit der
    Druckkammer oder dem Horn verbunden sein darf. Membran = von Hand markiert oder das eingezeichnete Bauteil, das der Druckkammer
    am nächsten liegt. Ihre Seiten trennt die Ebene durch den Schwerpunkt der Membranzellen (Normale = kleinste Hauptachse):
    Luftzellen an der Membran auf der Seite des Druckkammer-Punkts sind vorn, die anderen hinten."""
    fid = raster.region_id(e['druckkammer'])
    dF = _dilate(lbl == fid)
    wahl, hinweis = getattr(raster, 'membran_hand', {}).get(i), ' (von Hand markiert)'
    if wahl is None and nmem:
        kand = [k for k in np.unique(mem_lbl[dF & (mem_lbl > 0)]) if k > 0]
        if kand:
            zc = np.array(e['druckkammer']) / raster.h - 0.5
            cms = ndimage.center_of_mass(mem_lbl > 0, mem_lbl, kand)
            wahl = mem_lbl == kand[int(np.argmin([np.sum((np.array(c) - zc) ** 2) for c in cms]))]
            hinweis = ' (diaphragm from the model)'
    if wahl is None or not wahl.any():
        raise ValueError('Driver %d: an open rear needs the diaphragm – place it over the driver opening with the "Diaphragm" tool '
                         'or draw the driver in the model (part "Membran").' % (i + 1))
    pts = (np.argwhere(wahl) + 0.5) * raster.h
    c = pts.mean(axis=0)
    n = np.linalg.eigh(np.cov((pts - c).T) + 1e-12 * np.eye(3))[1][:, 0]
    n = n * (np.sign(np.dot(np.asarray(e['druckkammer'], float) - c, n)) or 1.0)  # zeigt jetzt zur Druckkammer
    # Luftflächen der Membran; vorn/hinten nach der Richtung der Fläche (Stirnkanten der Membran zählen nicht)
    zellen, ax, sd = flaechen_zu(_dilate(wahl) & ~wahl & (lbl > 0), wahl)
    raus = np.where(sd == 1, -1.0, 1.0) * n[ax]  # Flächennormale von der Membran zur Luft, Anteil in Richtung n
    # Nur Flächen zu den Lufträumen, die gerechnet werden: vorn die Druckkammer, hinten der größte Luftraum hinter der Membran.
    # Kleine eingeschlossene Taschen an der Membran (z. B. im feinen Raster zwischen Membran und Schallwand) zählen nicht.
    reg = lbl[tuple(zellen.T)]
    vor, hin = (raus > 0.3) & (reg == fid), raus < -0.3
    if hin.any():
        ids, anz = np.unique(reg[hin], return_counts=True)
        rid = int(ids[np.argmax(anz)])
        hin &= reg == rid
    if not vor.any() or not hin.any():
        raise ValueError('Driver %d: there is not air on both sides of the diaphragm (front to the pressure chamber, rear free)' % (i + 1))
    ff, fr = (zellen[vor], ax[vor], sd[vor]), (zellen[hin], ax[hin], sd[hin])
    info = {'druckkammer_region': fid, 'rueckkammer_region': rid, 'rueckseite_offen': True,
            'membranflaeche_vorn_cm2': len(ff[0]) * raster.h ** 2 * 1e4,
            'membranflaeche_hinten_cm2': len(fr[0]) * raster.h ** 2 * 1e4}
    if log:
        log('  Driver %d: open rear, diaphragm %d/%d faces (front/rear)%s, rear air space %d%s' % (
            i + 1, len(ff[0]), len(fr[0]), hinweis, rid, ' (same as front)' if rid == fid else ''))
    return {'F': fid, 'R': rid, 'vorn': ff, 'hinten': fr, 'info': info}


def einbauten(raster, einbau, log=print):
    lbl, _ = raster.regionen()
    mem = raster.masken.get('membran')
    mem_lbl, nmem = ndimage.label(mem, structure=np.ones((3, 3, 3))) if mem is not None and mem.any() else (None, 0)
    out, belegt = [], {}
    for i, e in enumerate(einbau):
        if e.get('rueck_offen') and e.get('rueckkammer') is None:
            out.append(offene_rueckseite(raster, i, e, lbl, mem_lbl, nmem, log))
            continue
        fid = raster.region_id(e['druckkammer'])
        rid = raster.region_id(e['rueckkammer'])
        if fid == rid:
            leck = raster.leckweg(e['druckkammer'], e['rueckkammer'])
            txt = 'Driver %d: pressure chamber and rear chamber are connected (no tight seal).' % (i + 1)
            if leck:
                for s in leck['engstellen']:
                    txt += '\n  Narrow point at %s, width about %.1f cm, parts: %s' % (
                        voxel.fmt_punkt(s['punkt']), 100 * s['weite'], ', '.join(s['teile']) or '–')
                txt += '\n  Fix: place a seal (axis-aligned box) over the gap.'
            raise ValueError(txt)
        F, R = lbl == fid, lbl == rid
        dF, dR = _dilate(F), _dilate(R)
        wahl, hinweis = None, ''
        hand = getattr(raster, 'membran_hand', {}).get(i)
        if hand is not None:
            wahl, hinweis = hand, ' (marked by hand)'
            if not ((hand & dF).any() and (hand & dR).any()):
                raise ValueError('Driver %d: the marked diaphragm does not lie between the pressure and rear chamber' % (i + 1))
        elif nmem:
            kand = np.unique(mem_lbl[dF & dR & (mem_lbl > 0)])
            kand = [k for k in kand if k > 0]
            if isinstance(e.get('membran'), str):
                kand = [k for k in kand if _teil_in(raster, mem_lbl == k, e['membran'])]
            if len(kand) > 1:
                ziel = e.get('membran') if isinstance(e.get('membran'), (list, tuple)) else \
                    (np.array(e['druckkammer']) + np.array(e['rueckkammer'])) / 2
                zc = np.array(ziel) / raster.h - 0.5
                cms = ndimage.center_of_mass(mem, mem_lbl, kand)
                kand = [kand[int(np.argmin([np.sum((np.array(c) - zc) ** 2) for c in cms]))]]
                hinweis = ' (diaphragm nearest to the point chosen)'
            if kand:
                wahl = mem_lbl == kand[0]
                if kand[0] in belegt:
                    raise ValueError('Drivers %d and %d share the same diaphragm – please specify "membran"'
                                     % (belegt[kand[0]] + 1, i + 1))
                belegt[kand[0]] = i
        if wahl is None:
            # no diaphragm drawn: use the separating wall cells between both air spaces
            wahl = (raster.lab != voxel.LUFT) & dF & dR
            hinweis = ' (no diaphragm in the model, separating cells used as diaphragm)'
            if not wahl.any():
                raise ValueError('Driver %d: no diaphragm found between the pressure and rear chamber. Draw the '
                                 'driver (part "Membran") or close the driver hole with a seal one cell '
                                 'thick.' % (i + 1))
        ff = flaechen_zu(F, wahl)
        fr = flaechen_zu(R, wahl)
        if not len(ff[0]) or not len(fr[0]):
            raise ValueError('Driver %d: the diaphragm does not border both air spaces' % (i + 1))
        info = {'druckkammer_region': fid, 'rueckkammer_region': rid,
                'membranflaeche_vorn_cm2': len(ff[0]) * raster.h ** 2 * 1e4,
                'membranflaeche_hinten_cm2': len(fr[0]) * raster.h ** 2 * 1e4}
        mag = raster.masken.get('magnet')
        if mag is not None and mag.any():
            near = ndimage.binary_dilation(wahl, iterations=int(np.ceil(0.15 / raster.h)))
            mg = mag & near
            if mg.any():
                nf, nr = (mg & dF).sum(), (mg & dR).sum()
                info['magnetseite'] = 'druckkammer' if nf > nr else 'rueckkammer'
        if log:
            log('  Driver %d: diaphragm %d/%d faces (front/rear)%s%s' % (
                i + 1, len(ff[0]), len(fr[0]), hinweis,
                ', magnet side facing the ' + ('pressure chamber' if info.get('magnetseite') == 'druckkammer' else 'rear chamber')
                if 'magnetseite' in info else ''))
        out.append({'F': fid, 'R': rid, 'vorn': ff, 'hinten': fr, 'info': info})
    return out


def _teil_in(raster, maske, muster):
    ids = np.unique(raster.teil[maske])
    return any(i >= 0 and muster.lower() in raster.modell.parts[i].lower() for i in ids)


# ---------- Problem aufbauen ----------

KLEIN_CM2 = 100      # Öffnungen darunter (und unter KLEIN_ANTEIL der größten) gelten als zu,
KLEIN_ANTEIL = 0.02  # z. B. Buchsen- oder Kabelausschnitte, die im Modell offen, in echt aber verschlossen sind


def mundflaechen(raster, domain, lbl, kleine_zu=True):
    """Flächen der Luftzellen auf dem Rand des Rasters, je Ebene (achse, seite) und Luftraum.
    kleine_zu: kleine einzelne Öffnungen bleiben schallhart (geschlossen).
    → (out, geschlossen): out[(a, s)] = [{region, zellen, uv}], geschlossen = [{seite, flaeche_cm2}]"""
    teile = []
    for a in range(3):
        for s in (0, 1):
            sl = [slice(None)] * 3
            sl[a] = 0 if s == 0 else raster.dims[a] - 1
            lab2, n = ndimage.label(domain[tuple(sl)])
            for k in range(1, n + 1):
                ij = np.argwhere(lab2 == k)
                c = np.insert(ij, a, sl[a], axis=1)
                teile.append({'a': a, 's': s, 'zellen': c, 'region': int(lbl[tuple(c[0])]),
                              'flaeche_cm2': len(c) * raster.h ** 2 * 1e4})
    groesste = max((t['flaeche_cm2'] for t in teile), default=0)
    out, geschlossen = {}, []
    for t in teile:
        a, s = t['a'], t['s']
        if kleine_zu and t['flaeche_cm2'] < min(KLEIN_CM2, KLEIN_ANTEIL * groesste):
            geschlossen.append({'seite': next(q['name'] for q in voxel.SEITEN if q['a'] == a and q['side'] == s),
                                'flaeche_cm2': round(t['flaeche_cm2'], 1)})
            continue
        lst = out.setdefault((a, s), [])
        alt = next((x for x in lst if x['region'] == t['region']), None)
        if alt:
            alt['zellen'] = np.concatenate([alt['zellen'], t['zellen']])
        else:
            lst.append({'region': t['region'], 'zellen': t['zellen']})
    for (a, s), lst in out.items():
        u, v = (a + 1) % 3, (a + 2) % 3
        for x in lst:
            x['uv'] = (x['zellen'][:, [u, v]] + 0.5) * raster.h
    return out, geschlossen


def baue_problem(raster, regionen, quellen, patch, log=print, mund=None):
    """regionen: Region-IDs (Lufträume) des Rechengebiets; quellen: [(vorn, hinten)] mit Flächen-Tupeln
    (zellen, achse, seite); hinten darf None sein. mund: Ergebnis von mundflaechen (sonst neu bestimmt)."""
    lbl, _ = raster.regionen()
    domain = np.isin(lbl, list(regionen))
    if mund is None:
        mund, _ = mundflaechen(raster, domain, lbl)
    t0 = time.time()
    netz = fem.Netz(raster, domain)
    K, M = netz.matrizen()
    komp = np.zeros(netz.ndof, dtype=np.int64)
    el_reg = lbl[tuple(netz.cells.T)]
    for c in range(8):
        komp[netz.dof[:, c]] = el_reg
    if log:
        log('  Mesh: %d cells, %d degrees of freedom, air volume %.1f L (%.1f s)'
            % (netz.ne, netz.ndof, netz.volumen * 1e3, time.time() - t0))
    S = np.zeros((netz.ndof, len(quellen)))
    for j, (vorn, hinten) in enumerate(quellen):
        for fl, sg in ((vorn, 1.0), (hinten, -1.0)):
            if fl is None:
                continue
            ff = netz.flaechen(fl[0], fl[1], fl[2])
            w = netz.flaechenlast(ff)
            S[:, j] += sg * w / w.sum()
    ebenen, Bs, pfl, mund_info = [], [], [], []
    for (a, s), lst in sorted(mund.items()):
        e = strahlung.Ebene([strahlung.Strahler(x['uv'], raster.h, name='Region %d' % x['region']) for x in lst], patch)
        cells = np.concatenate([x['zellen'] for x in lst])
        ff = netz.flaechen(cells, a, s)
        Bs.append(netz.patchmatrix(ff, e.patch, e.npatch))
        pfl.append(e.patch_flaeche)
        ebenen.append(e)
        seitenname = next(q['name'] for q in voxel.SEITEN if q['a'] == a and q['side'] == s)
        for x in lst:
            mund_info.append({'seite': seitenname, 'region': x['region'], 'flaeche_cm2': len(x['uv']) * raster.h ** 2 * 1e4,
                              'rechteck_cm': [float(x['uv'][:, 0].min() * 100 - raster.h * 50), float(x['uv'][:, 0].max() * 100 + raster.h * 50),
                                              float(x['uv'][:, 1].min() * 100 - raster.h * 50), float(x['uv'][:, 1].max() * 100 + raster.h * 50)]})
        if log:
            log('  Radiation %s: %d faces, %d patches (%.0f cm), subcells %d'
                % (mund_info[-1]['seite'], e.nfl, e.npatch, e.m * raster.h * 100, e.nsub))
    import scipy.sparse as sp
    B = sp.hstack(Bs).tocsr() if Bs else sp.csr_matrix((netz.ndof, 0))
    prob = fem.Problem(K, M, S, B, ebenen, komp, np.concatenate(pfl) if pfl else np.zeros(0))
    return prob, netz, mund_info


def loese(prob, freqs, methode='auto', direkt_bis=3000, log=print):
    t0 = time.time()
    if methode == 'direkt' or (methode == 'auto' and prob.K.shape[0] <= direkt_bis):
        Z, H = prob.direkt(freqs)
        art = 'direkt'
    else:
        prob.reduziere(max(freqs), fmin=min(freqs), log=log)
        Z, H = prob.reduziert(freqs)
        art = 'reduziert'
    if log:
        log('  Solution (%s): %d frequencies in %.1f s' % (art, len(freqs), time.time() - t0))
    return Z, H, art


# ---------- Auftrag ----------

PROJEKT = os.path.dirname(HIER)


def finde_datei(name, basis, extra=()):
    """Datei suchen: wie angegeben (relativ zu basis), sonst nach Dateinamen in basis, den Ordnern in extra,
    datenblaetter/ und zuletzt im ganzen Projektordner. Die App kennt nur den Dateinamen, nicht den Pfad."""
    kandidaten = [name if os.path.isabs(name) else os.path.join(basis, name)]
    if not os.path.isabs(name):
        kandidaten.append(os.path.join(PROJEKT, name))  # Pfad relativ zum Projektordner, z. B. bibliothek/boxen/<Box>/<Datei>.dae
    datei = os.path.basename(name)
    for d in (basis,) + tuple(extra) + (os.path.join(PROJEKT, 'datenblaetter'), PROJEKT):
        kandidaten.append(os.path.join(d, datei))
    for k in kandidaten:
        if os.path.isfile(k):
            return os.path.normpath(k)
    for wurzel, ordner, dateien in os.walk(PROJEKT):
        ordner[:] = [o for o in ordner if not o.startswith('.') and o != '__pycache__']
        if datei in dateien:
            return os.path.join(wurzel, datei)
    raise FileNotFoundError('%s not found (searched next to the job, in datenblaetter/ and in the project folder)' % datei)


def finde_auftrag(arg):
    """Job as given, otherwise in werkzeug/auftraege/ or in the Downloads folder (where the app saves it)."""
    if os.path.isfile(arg):
        return os.path.abspath(arg)
    downloads = os.path.join(os.path.expanduser('~'), 'Downloads')
    for d in (os.path.join(HIER, 'auftraege'), os.path.join(PROJEKT, 'Input'), downloads):
        k = os.path.join(d, os.path.basename(arg))
        if os.path.isfile(k):
            return k
    raise FileNotFoundError('Job %s not found (neither in werkzeug/auftraege/ nor in %s)' % (arg, downloads))


def _treiber_daten(auftrag, basis):
    t = auftrag.get('treiber')
    if isinstance(t, str):
        with open(finde_datei(t, basis, (os.path.join(PROJEKT, 'bibliothek', 'treiber'),)), encoding='utf-8') as fh:
            t = json.load(fh).get('treiber')
    if isinstance(t, dict):
        t = [t]
    if not t:
        raise ValueError('The job has no driver data ("treiber")')
    return t


def baue_raster(modell, auftrag, log=print):
    r = voxel.Raster(modell, auftrag.get('aufloesung', 0.01))
    for ab in auftrag.get('abdichtungen', []):
        n = r.abdichtung(ab['achse'], ab['von'], ab['bis'], rechteck=ab.get('rechteck'),
                         polygon=ab.get('polygon'), aussparung=ab.get('aussparung'))
        if log:
            log('  Seal %s: %d cells' % (ab.get('name', ''), n))
    # Membran von Hand (Treiber nicht im Modell): Quader wird dicht und gilt als Membran dieses Treibers
    r.membran_hand = {}
    for i, e in enumerate(auftrag.get('einbau', [])):
        mp = e.get('membran_prisma')
        if mp:
            n, r.membran_hand[i] = r.abdichtung(mp['achse'], mp['von'], mp['bis'], rechteck=mp.get('rechteck'),
                                                polygon=mp.get('polygon'), mit_maske=True)
            if log:
                log('  Diaphragm of driver %d by hand: %d cells' % (i + 1, n))
    return r


def spitzenspeicher_mb():
    """Peak memory of this process so far (MB), if it can be determined."""
    try:
        import ctypes
        from ctypes import wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD)] + [
                (n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                                'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                                                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wt.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return None
        return round(pmc.PeakWorkingSetSize / 2 ** 20)
    except Exception:
        try:
            import resource
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        except Exception:
            return None


def rechne(auftrag, basis='.', log=print, modell=None):
    """Calculate a job → (result like sim.ergebnis, info, model). modell: an already loaded model (tests)."""
    t_start = time.time()
    if log is None:
        log = lambda *a: None  # noqa: E731
    if modell is None and auftrag.get('dae_text'):
        # The app puts the model straight into the job (then no second file is needed)
        log('Model: %s (included in the job)' % auftrag.get('dae', ''))
        modell = dae.parse(auftrag['dae_text'])
    if modell is None:
        pfad = finde_datei(auftrag.get('dae_pfad') or auftrag['dae'], basis)
        log('Model: %s' % pfad)
        modell = dae.parse(pfad)
    log('  %d triangles, %.1f × %.1f × %.1f cm' % ((len(modell.tri),) + tuple(modell.size * 100)))
    mund = voxel.find_mouth(modell)
    if mund:
        log('  Mouth (ray tracing): %s, %.0f cm²' % (mund[0]['name'], mund[0]['flaeche'] * 1e4))
    t0 = time.time()
    raster = baue_raster(modell, auftrag, log)
    log('  Grid %s cells of %.1f cm (%.1f s)' % ('×'.join(map(str, raster.dims)), raster.h * 100, time.time() - t0))
    eb = einbauten(raster, auftrag['einbau'], log)
    lbl, _ = raster.regionen()
    regionen = sorted({e['F'] for e in eb} | {e['R'] for e in eb})
    oeffnungen, geschlossen = mundflaechen(raster, np.isin(lbl, regionen), lbl, auftrag.get('kleine_oeffnungen_zu', True))
    for g in geschlossen:
        log('  Small opening %s, %.0f cm²: calculated as closed' % (g['seite'], g['flaeche_cm2']))
    mit_mund = {x['region'] for lst in oeffnungen.values() for x in lst}
    kammern = {}
    for rid in regionen:
        m = lbl == rid
        kammern[rid] = {'volumen_L': round(raster.volumen(m) * 1e3, 2), 'offen': rid in mit_mund,
                        'treiber_vorn': [i + 1 for i, e in enumerate(eb) if e['F'] == rid],
                        'treiber_hinten': [i + 1 for i, e in enumerate(eb) if e['R'] == rid]}
        log('  Air space %d: %.2f L%s, pressure chamber of %s, rear chamber of %s' % (
            rid, kammern[rid]['volumen_L'], ' (open to the outside)' if kammern[rid]['offen'] else '',
            kammern[rid]['treiber_vorn'] or '–', kammern[rid]['treiber_hinten'] or '–'))
    warn = []
    for rid, k in kammern.items():
        gewollt = all(eb[t - 1]['info'].get('rueckseite_offen') for t in k['treiber_hinten'])  # Paraflex & co.: built that way
        if k['offen'] and k['treiber_hinten'] and not k['treiber_vorn'] and not gewollt:
            warn.append('Rear chamber (air space %d) is open to the outside – leak or bass reflex?' % rid)
    fr = hornsim1d.log_grid(auftrag.get('fmin', 20), auftrag.get('fmax', 200), auftrag.get('punkte_pro_oktave', 12))
    patch = auftrag.get('mund_patch') or float(np.clip(C / (12 * max(fr)), 2 * raster.h, 0.25))
    prob, netz, mund_info = baue_problem(raster, regionen, [(e['vorn'], e['hinten']) for e in eb], patch, log, oeffnungen)
    if len(mund_info) > 1:
        warn.append('Radiation through several openings: ' + '; '.join(
            '%s %.0f cm²' % (m['seite'], m['flaeche_cm2']) for m in mund_info))
    Z, H, art = loese(prob, fr, auftrag.get('methode', 'auto'), log=log)
    tdat = _treiber_daten(auftrag, basis)
    td = [tdat[i] if i < len(tdat) else tdat[0] for i in range(len(eb))]
    for t in td:
        miss = tr.fehlend(t)
        if miss:
            raise ValueError('Driver data incomplete: ' + ', '.join(miss))
    pol = [e.get('polaritaet', 1) for e in auftrag['einbau']]
    erg = tr.antwort(fr, Z, H, [tr.si(t) for t in td], pol, auftrag.get('verschaltung', 'parallel'))
    for w in warn:
        log('Note: ' + w)
    info = {
        'aufloesung_cm': raster.h * 100, 'zellen': int(netz.ne), 'freiheitsgrade': int(netz.ndof),
        'methode': art, 'rechenzeit_s': round(time.time() - t_start, 1), 'mund_patch_cm': round(patch * 100, 1),
        'speicher_hauptprozess_mb': spitzenspeicher_mb(),
        'luftraeume': {str(k): v for k, v in kammern.items()}, 'treiber': [e['info'] for e in eb],
        'abstrahlung': mund_info, 'hinweise': warn, 'geschlossen': geschlossen,
        'mund_strahlverfolgung': ({'seite': mund[0]['name'], 'flaeche_cm2': round(mund[0]['flaeche'] * 1e4)}
                                  if mund else None),
    }
    return erg, info, modell


def ergebnis3d(erg, info, treibername):
    r = lambda arr, d: [None if v is None else round(float(v), d) for v in arr]  # noqa: E731
    mf = sum(m['flaeche_cm2'] for m in info['abstrahlung'])
    return {'f': r(erg['f'], 3), 'db': r(erg['db'], 2), 'ph': r(erg['ph'], 4), 'x': r(erg['x'], 3),
            'zin': r(erg['zin'], 2), 'maxspl': r(erg['maxspl'], 2),
            'x_treiber': [r(x, 3) for x in erg['x_treiber']],
            'mundflaeche': round(mf), 'zeit': int(time.time() * 1000), 'treiber': treibername,
            'quelle': 'werkzeug/rs3d.py (3D-FEM)', 'info': info}


def kennwerte(f, db):
    """Mittelpegel 50–100 Hz (Leistungsmittel) und untere −6-dB-Grenze: erste Frequenz von unten, ab der
    der Pegel das Mittel − 6 dB erreicht (spätere Senken darüber zählen nicht)."""
    f, db = np.asarray(f, float), np.asarray(db, float)
    sel = (f >= 50) & (f <= 100)
    mittel = 10 * np.log10(np.mean(10 ** (db[sel] / 10))) if sel.any() else np.nan
    ziel = mittel - 6
    f6 = None
    hit = np.nonzero(db >= ziel)[0]
    if len(hit) and hit[0] > 0:
        k = hit[0]
        t = (ziel - db[k - 1]) / (db[k] - db[k - 1])
        f6 = float(f[k - 1] * (f[k] / f[k - 1]) ** t)
    return {'mittel_50_100_db': round(float(mittel), 2), 'f_minus6_hz': None if f6 is None else round(f6, 1)}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    import gpu
    if '--cpu' in argv:
        argv.remove('--cpu')
        gpu.abschalten('rs3d.py --cpu')
    if not argv:
        print(__doc__)
        return 2
    apfad = finde_auftrag(argv[0])
    basis = os.path.dirname(apfad)
    print('Job: %s' % apfad)
    ok, name = gpu.verfuegbar()
    print('Running on: %s' % ('GPU ' + name if ok else 'CPU (GPU: %s)' % name))
    with open(apfad, encoding='utf-8') as fh:
        auftrag = json.load(fh)
    erg, info, modell = rechne(auftrag, basis)
    tdat = _treiber_daten(auftrag, basis)
    vorlage = auftrag.get('box')
    if isinstance(vorlage, str):  # path; from the app the box comes directly as an object
        with open(finde_datei(vorlage, basis, (os.path.join(PROJEKT, 'bibliothek', 'boxen'),)), encoding='utf-8') as fh:
            vorlage = json.load(fh)
    if vorlage:
        box = copy.deepcopy(vorlage)
    else:
        stem = os.path.splitext(os.path.basename(auftrag['dae']))[0]
        name = auftrag.get('name', stem)
        box = {'typ': 'rs-box', 'version': 1, 'profil': {
            'id': name.lower().replace(' ', '-'), 'name': name, 'kategorie': 'sub', 'bauform': '3D-Modell',
            'treiberId': tdat[0].get('id'),
            'masse': {'b': round(modell.size[0] * 100, 1), 'h': round(modell.size[2] * 100, 1), 't': round(modell.size[1] * 100, 1)},
            'gewicht': None, 'mund': None, 'pe': None, 'impedanz': None, 'fmin': auftrag.get('fmin', 20),
            'fmax': auftrag.get('fmax', 200), 'notiz': '', 'dae': {'datei': os.path.basename(auftrag['dae']),
                                                                    'masse': [round(v, 4) for v in modell.size]},
            'messungen': [], 'messAbstand': 1, 'messBoxen': 1, 'glaettung': 6,
            'sim': {'typ': 'flh', 'treiberAnzahl': len(auftrag['einbau']), 'verschaltung': auftrag.get('verschaltung', 'parallel'),
                    'segmente': [], 'ergebnis': None}, 'planerKurve': 'sim'}, 'treiber': tdat[:1]}
    prof = box['profil']
    prof.setdefault('sim', {})
    e3 = ergebnis3d(erg, info, tdat[0].get('name', ''))
    e3['info']['kennwerte'] = kennwerte(erg['f'], erg['db'])
    # What it was calculated from, so the app can recognize an outdated result
    e3['auftrag'] = {'datei': os.path.basename(apfad), 'dae': os.path.basename(auftrag['dae']),
                     'erstellt': auftrag.get('erstellt'), 'marker_stand': auftrag.get('marker_stand')}
    # Comparison with the 1D calculation, if the template has horn segments or is a direct radiator
    sim = prof['sim']
    if sim.get('segmente') or sim.get('typ') in ('geschlossen', 'reflex'):
        try:
            r1 = hornsim1d.simulate(sim, tdat[0], erg['f'])
            e3['info']['vergleich1d'] = {'kennwerte': kennwerte(r1['f'], r1['db']),
                                         'db': [round(v, 2) for v in r1['db']]}
        except Exception as ex:  # the comparison is only an extra
            e3['info']['vergleich1d'] = {'fehler': str(ex)}
    sim['ergebnis3d'] = e3
    ziel = auftrag.get('ausgabe')
    if ziel:
        ziel = os.path.join(basis, ziel)
    else:
        stem = os.path.basename(apfad)
        for suf in ('.json', '.auftrag'):
            if stem.endswith(suf):
                stem = stem[:-len(suf)]
        # RS3D_AUSGABE: target folder, e.g. the Output folder of the standalone 3D Simulator (otherwise werkzeug/ausgabe/)
        ziel = os.path.join(os.environ.get('RS3D_AUSGABE') or os.path.join(HIER, 'ausgabe'), stem + '.box.json')
    os.makedirs(os.path.dirname(ziel), exist_ok=True)
    with open(ziel, 'w', encoding='utf-8') as fh:
        json.dump(box, fh, ensure_ascii=False, indent=1)
    k = e3['info']['kennwerte']
    print('3D: %.1f dB mean 50–100 Hz, −6 dB at %s Hz' % (k['mittel_50_100_db'], k['f_minus6_hz']))
    if 'vergleich1d' in e3['info'] and 'kennwerte' in e3['info']['vergleich1d']:
        k1 = e3['info']['vergleich1d']['kennwerte']
        print('1D: %.1f dB mean 50–100 Hz, −6 dB at %s Hz' % (k1['mittel_50_100_db'], k1['f_minus6_hz']))
    print('Written: %s (%.0f s, main process up to %s MB)' % (os.path.normpath(ziel), e3['info']['rechenzeit_s'],
                                                                e3['info']['speicher_hauptprozess_mb']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
