"""Helmholtz-FEM auf dem Voxel-Hexaedernetz der Luftzellen (trilineare Elemente, scikit-fem).

Netz: jede Luftzelle ist ein Würfel-Element (Kante h). Knoten werden nur über Flächen geteilt: an jedem
Gitterpunkt bekommen die Luftzellen der 2×2×2-Umgebung, die über Flächen zusammenhängen, einen gemeinsamen
Knoten, sonst getrennte. So koppeln Lufträume nicht über Kanten/Ecken dünner, schräger Wände.

Elementmatrizen kommen aus scikit-fem (Referenzwürfel, ElementHex1); weil alle Elemente gleich groß sind,
wird damit vektorisiert assembliert. Rasterkorrektur je Element: Volumen ·(1 + 0,5·Wandflächen),
Leitfähigkeit in Achse a ·(1 + 0,5·Wandflächen quer zu a).

Schwache Form mit Druck p (e^{jωt}):  ∫∇p·∇q − k²∫pq + jωρ∫_Γ v_n q = 0   (v_n nach außen)
  Membran (Volumenfluss U, gleichmäßig verteilt):  rechte Seite jωρ·U·w,  w = ∫_Γm φ / A_m
  Mund (Patch P, Schnelle v_P):  Kopplung B λ mit λ_P = jωρ v_P und Bᵀp − G(k) λ = 0  (strahlung.py)
Gesamtsystem je Frequenz:  [K − k²M   B ] [p]   [jωρ S U]
                           [Bᵀ       −G ] [λ] = [   0   ]
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from strahlung import RHO, C

# ---------- Referenzelement aus scikit-fem ----------
_REF = None


def referenz():
    """Kx, Ky, Kz, M (8×8) des Einheitswürfels; lokale Ecke c = cx + 2·cy + 4·cz."""
    global _REF
    if _REF is None:
        from skfem import MeshHex, ElementHex1, Basis, BilinearForm, asm
        m = MeshHex()
        basis = Basis(m, ElementHex1(), intorder=4)
        mats = []
        for a in range(3):
            form = BilinearForm(lambda u, v, w, a=a: u.grad[a] * v.grad[a])
            mats.append(asm(form, basis).toarray())
        mats.append(asm(BilinearForm(lambda u, v, w: u * v), basis).toarray())
        # Knotenreihenfolge der skfem-Elementknoten → eigene Eckennummer
        p = m.p.T.round().astype(int)
        mine = p[:, 0] + 2 * p[:, 1] + 4 * p[:, 2]
        perm = np.argsort(mine)
        _REF = [M[np.ix_(perm, perm)] for M in mats]
    return _REF


def _lut():
    """Für jedes Belegungsmuster (8 Bit) der 2×2×2-Zellen um einen Gitterpunkt: Komponente je Zelle."""
    lut = np.full((256, 8), -1, dtype=np.int8)
    for pat in range(256):
        comp = list(range(8))

        def find(i):
            while comp[i] != i:
                i = comp[i]
            return i
        for q in range(8):
            if not (pat >> q) & 1:
                continue
            for b in (1, 2, 4):
                r = q ^ b
                if (pat >> r) & 1:
                    ra, rb = find(q), find(r)
                    if ra != rb:
                        comp[max(ra, rb)] = min(ra, rb)
        for q in range(8):
            if (pat >> q) & 1:
                lut[pat, q] = find(q)
    return lut


_LUT = _lut()
FACE_CORNERS = [[[c for c in range(8) if ((c >> a) & 1) == s] for s in (0, 1)] for a in range(3)]


class Netz:
    """Hexaedernetz aus einer Zellmaske (bool, dims) eines Rasters."""

    def __init__(self, raster, maske, wandkorrektur=True):
        t0 = time.time()
        self.h = h = raster.h
        dims = raster.dims
        self.dims = dims
        self.cells = np.argwhere(maske)                      # (ne, 3)
        ne = len(self.cells)
        self.ne = ne
        self.elem_of = np.full(dims, -1, dtype=np.int64)
        self.elem_of[tuple(self.cells.T)] = np.arange(ne)
        # Belegungsmuster je Gitterpunkt
        P = np.zeros(tuple(d + 2 for d in dims), dtype=np.uint8)
        P[1:-1, 1:-1, 1:-1] = maske
        vd = tuple(d + 1 for d in dims)
        pat = np.zeros(vd, dtype=np.uint8)
        for q in range(8):
            qx, qy, qz = q & 1, (q >> 1) & 1, (q >> 2) & 1
            pat |= (P[qx:qx + vd[0], qy:qy + vd[1], qz:qz + vd[2]] << q).astype(np.uint8)
        keys = np.empty((ne, 8), dtype=np.int64)
        for c in range(8):
            off = np.array([c & 1, (c >> 1) & 1, (c >> 2) & 1])
            v = self.cells + off
            lin = np.ravel_multi_index(v.T, vd)
            lab = _LUT[pat.flat[lin], 7 - c]
            keys[:, c] = lin * 8 + lab
        del pat, P
        uniq, inv = np.unique(keys.ravel(), return_inverse=True)
        self.dof = inv.reshape(ne, 8).astype(np.int64)
        self.ndof = len(uniq)
        vlin = uniq // 8
        self.knoten = np.stack(np.unravel_index(vlin, vd), axis=1) * h     # Koordinaten (ndof, 3)
        # Rasterkorrektur
        if wandkorrektur:
            nw = raster.wandnachbarn()[(slice(None),) + tuple(self.cells.T)].T.astype(float)  # (ne, 3)
        else:
            nw = np.zeros((ne, 3))
        self.cm = 1 + 0.5 * nw.sum(axis=1)
        self.D = np.stack([1 + 0.5 * (nw[:, 1] + nw[:, 2]), 1 + 0.5 * (nw[:, 0] + nw[:, 2]),
                           1 + 0.5 * (nw[:, 0] + nw[:, 1])], axis=1)
        self.volumen = float(self.cm.sum() * h ** 3)
        self.zeit = time.time() - t0

    def matrizen(self, block=200000):
        """Steifigkeit K und Masse M (CSR, ndof × ndof)."""
        Kx, Ky, Kz, M0 = referenz()
        h, n = self.h, self.ndof
        K = sp.csr_matrix((n, n))
        M = sp.csr_matrix((n, n))
        for s in range(0, self.ne, block):
            e = slice(s, min(self.ne, s + block))
            dof, D, cm = self.dof[e], self.D[e], self.cm[e]
            r = np.repeat(dof, 8, axis=1).ravel()
            c = np.tile(dof, (1, 8)).ravel()
            kv = (h * (D[:, 0, None, None] * Kx + D[:, 1, None, None] * Ky + D[:, 2, None, None] * Kz)).reshape(-1)
            mv = (h ** 3 * cm[:, None, None] * M0).reshape(-1)
            K = K + sp.csr_matrix((kv, (r, c)), shape=(n, n))
            M = M + sp.csr_matrix((mv, (r, c)), shape=(n, n))
        return K.tocsr(), M.tocsr()

    def flaechen(self, zellen, achse, seite):
        """Elementflächen: zellen (m, 3) Zellindizes, achse/seite je Fläche → (Element, Achse, Seite)."""
        e = self.elem_of[tuple(np.asarray(zellen).T)]
        if np.any(e < 0):
            raise ValueError('Face does not belong to any mesh cell')
        return np.stack([e, np.broadcast_to(achse, e.shape), np.broadcast_to(seite, e.shape)], axis=1)

    def flaechenlast(self, fl, gewicht=None):
        """Vektor ∫ φ_i dS über die Flächen fl (m, 3); gewicht je Fläche (Standard 1)."""
        out = np.zeros(self.ndof)
        a4 = self.h * self.h / 4
        g = np.ones(len(fl)) if gewicht is None else np.asarray(gewicht)
        for a in range(3):
            for s in (0, 1):
                sel = (fl[:, 1] == a) & (fl[:, 2] == s)
                if not sel.any():
                    continue
                corners = FACE_CORNERS[a][s]
                d = self.dof[fl[sel, 0]][:, corners]
                np.add.at(out, d.ravel(), np.repeat(g[sel] * a4, 4))
        return out

    def patchmatrix(self, fl, patch, npatch):
        """B (ndof × npatch): B_iP = ∫_P φ_i dS."""
        rows, cols, vals = [], [], []
        a4 = self.h * self.h / 4
        for a in range(3):
            for s in (0, 1):
                sel = (fl[:, 1] == a) & (fl[:, 2] == s)
                if not sel.any():
                    continue
                d = self.dof[fl[sel, 0]][:, FACE_CORNERS[a][s]]
                rows.append(d.ravel())
                cols.append(np.repeat(patch[sel], 4))
                vals.append(np.full(d.size, a4))
        if not rows:
            return sp.csr_matrix((self.ndof, npatch))
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(self.ndof, npatch))


# ---------- Löser ----------

class Problem:
    """K, M, Quellen S (ndof × N), Mund-Kopplung B (ndof × npatch) und Ebenen (strahlung.Ebene).

    komp: Komponente (zusammenhängender Luftraum) je Freiheitsgrad, für die blockweise Reduktion."""

    def __init__(self, K, M, S, B, ebenen, komp, patch_flaeche):
        self.K, self.M = K, M
        self.S = np.asarray(S, float)
        self.B = sp.csr_matrix(B)
        self.ebenen = ebenen
        self.komp = komp
        self.patch_flaeche = np.asarray(patch_flaeche, float)
        self.npatch = self.B.shape[1]
        self.info = {}

    def G(self, k):
        Gs = [e.G(k) for e in self.ebenen]
        if not Gs:
            return np.zeros((0, 0), complex)
        out = np.zeros((self.npatch, self.npatch), complex)
        o = 0
        for g in Gs:
            n = g.shape[0]
            out[o:o + n, o:o + n] = g
            o += n
        return out

    def _auswerten(self, p_ports, lam, w):
        """Z (N×N) = Sᵀp je Einheits-Volumenfluss, H (N) = Fernfelddruck in 1 m je Einheits-Volumenfluss."""
        Z = p_ports
        H = (self.patch_flaeche @ lam) / (2 * np.pi) if lam.size else np.zeros(Z.shape[1], complex)
        return Z, H

    # ----- direkt: volles System je Frequenz -----
    def direkt(self, freqs, log=None):
        n, N = self.K.shape[0], self.S.shape[1]
        out_Z, out_H = [], []
        for f in freqs:
            k = 2 * np.pi * f / C
            w = 2 * np.pi * f
            A = (self.K - k * k * self.M).astype(complex)
            if self.npatch:
                G = self.G(k)
                big = sp.bmat([[A, self.B], [self.B.T, sp.csr_matrix(-G)]], format='csc')
            else:
                big = A.tocsc()
            rhs = np.zeros((big.shape[0], N), complex)
            rhs[:n] = 1j * w * RHO * self.S
            x = spla.splu(big, permc_spec='COLAMD').solve(rhs)
            Z, H = self._auswerten(self.S.T @ x[:n], x[n:], w)
            out_Z.append(Z)
            out_H.append(H)
        return np.array(out_Z), np.array(out_H)

    # ----- reduziert: Moden + statische Antworten je Komponente -----
    def reduziere(self, fmax, fmin=10.0, fcut_faktor=2.5, direkt_bis=8000, log=print, zweite_ordnung=True,
                  prozesse=None):
        """Reduktionsbasis je Komponente. fcut = fcut_faktor·fmax: alle Moden darunter werden aufgenommen."""
        kcut2 = (2 * np.pi * fcut_faktor * fmax / C) ** 2
        sigma = (2 * np.pi * fmin / C) ** 2
        self.basen = []
        P = sp.hstack([sp.csr_matrix(self.S), self.B]).tocsr()
        for c in np.unique(self.komp):
            idx = np.nonzero(self.komp == c)[0]
            Kc = self.K[idx][:, idx].tocsr()
            Mc = self.M[idx][:, idx].tocsr()
            Pc = P[idx]
            used = np.unique(Pc.nonzero()[1])
            Pc = Pc[:, used].toarray()
            t0 = time.time()
            n = len(idx)
            nS = int(np.sum(used < self.S.shape[1]))
            if n <= direkt_bis:
                lu = spla.splu((Kc + sigma * Mc).tocsc())
                modes = _moden_direkt(Kc, Mc, sigma, lu, kcut2)
                X = lu.solve(Pc) if Pc.size else np.zeros((n, 0))
                vecs = [modes, X]
                if zweite_ordnung and X.size:
                    vecs.append(lu.solve(Mc @ X))
                basis = np.hstack(vecs)
                nmod = modes.shape[1]
            else:
                basis, nmod = _basis_gross(Kc, Mc, sigma, kcut2, Pc, nS if zweite_ordnung else 0, log, prozesse)
            del Pc
            V = _orthonormal(basis, Mc)
            self.basen.append({'idx': idx, 'r': V.shape[1], 'Kr': _gram(V, Kc), 'Mr': _gram(V, Mc),
                               'Sr': V.T @ self.S[idx], 'Br': np.asarray((self.B[idx].T @ V).T)})
            del V, basis
            if log:
                log('  Air space %d: %d degrees of freedom, %d modes up to %.0f Hz, basis %d (%.1f s)'
                    % (c, n, nmod, fcut_faktor * fmax, self.basen[-1]['r'], time.time() - t0))

    def reduziert(self, freqs):
        N = self.S.shape[1]
        r = sum(b['r'] for b in self.basen)
        Kr = np.zeros((r, r))
        Mr = np.zeros((r, r))
        Sr = np.zeros((r, N))
        Br = np.zeros((r, self.npatch))
        o = 0
        for b in self.basen:
            m = b['r']
            Kr[o:o + m, o:o + m] = b['Kr']
            Mr[o:o + m, o:o + m] = b['Mr']
            Sr[o:o + m] = b['Sr']
            Br[o:o + m] = b['Br']
            o += m
        out_Z, out_H = [], []
        for f in freqs:
            k = 2 * np.pi * f / C
            w = 2 * np.pi * f
            G = self.G(k)
            A = np.block([[Kr - k * k * Mr, Br], [Br.T, -G]]) if self.npatch else Kr - k * k * Mr
            rhs = np.zeros((A.shape[0], N), complex)
            rhs[:r] = 1j * w * RHO * Sr
            x = np.linalg.solve(A, rhs)
            Z, H = self._auswerten(Sr.T @ x[:r], x[r:], w)
            out_Z.append(Z)
            out_H.append(H)
        return np.array(out_Z), np.array(out_H)


def _gram(X, A, blk=16):
    """Xᵀ·A·X blockweise (ohne volle Kopie von A·X)."""
    m = X.shape[1]
    G = np.empty((m, m))
    for j in range(0, m, blk):
        G[:, j:j + blk] = X.T @ (A @ X[:, j:j + blk])
    return 0.5 * (G + G.T)


def _rechts_mal(X, T, blk=65536):
    """X·T zeilenblockweise in X selbst (T hat höchstens so viele Spalten wie X); gibt die Sicht zurück."""
    r = T.shape[1]
    for i in range(0, X.shape[0], blk):
        X[i:i + blk, :r] = X[i:i + blk] @ T
    return X[:, :r]


def _orthonormal(X, M, tol=1e-10):
    """M-orthonormale Basis von span(X), fast abhängige Richtungen werden verworfen. X wird überschrieben."""
    if X.shape[1] == 0:
        return X
    X = np.ascontiguousarray(X)
    nrm = np.sqrt(np.maximum(np.diag(_gram(X, M)), 0))
    if not np.all(nrm > 0):
        X = X[:, nrm > 0]
        nrm = nrm[nrm > 0]
    X /= nrm
    ev, U = np.linalg.eigh(_gram(X, M))
    keep = ev > tol * ev.max()
    V = _rechts_mal(X, U[:, keep] / np.sqrt(ev[keep]))
    # eine Nachorthogonalisierung gegen Rundungsfehler
    L = np.linalg.cholesky(_gram(V, M))
    return _rechts_mal(V, np.linalg.inv(L).T)


def _moden_direkt(K, M, sigma, lu, kcut2):
    """Alle Eigenvektoren mit λ < kcut² (Shift-Invert-Lanczos mit LU von K + σM)."""
    n = K.shape[0]
    op = spla.LinearOperator((n, n), matvec=lu.solve, dtype=float)
    k = min(n - 2, 8)
    while True:
        vals, vecs = spla.eigsh(K, k=k, M=M, sigma=-sigma, which='LM', OPinv=op, tol=1e-10)
        if vals.max() >= kcut2 or k >= n - 2 or k >= 400:
            return vecs[:, vals < kcut2]
        k = min(n - 2, 2 * k + 4)


def _basis_gross(K, M, sigma, kcut2, Pc, nS, log, prozesse=None):
    """Basis für große Lufträume (AMG statt LU):
    1. Moden auf der ersten AMG-Grobstufe (P0ᵀ·K·P0, P0ᵀ·M·P0), auf das feine Netz übertragen,
    2. je ein Schritt inverse Iteration (K+σM)⁻¹·M·x auf dem feinen Netz (macht die Moden genau),
    3. statische Antworten auf alle Anschlüsse (Membranen, Mundfelder), für die Membranen zweite Ordnung.
    Gelöst wird mit AMG-vorkonditioniertem CG, bei vielen Spalten parallel in mehreren Prozessen.
    Gibt (Basis n × m, Zahl der Moden) zurück."""
    import pyamg
    t0 = time.time()
    n = K.shape[0]
    A = (K + sigma * M).tocsr()
    ml = pyamg.smoothed_aggregation_solver(A, B=np.ones((n, 1)), **AMG_OPT)
    if len(ml.levels) < 2:   # zu klein für eine Grobstufe: direkt
        lu = spla.splu(A.tocsc())
        modes = _moden_direkt(K, M, sigma, lu, kcut2)
        X = lu.solve(Pc) if Pc.size else np.zeros((n, 0))
        return np.hstack([modes, X] + ([lu.solve(M @ X[:, :nS])] if nS else [])), modes.shape[1]
    P0 = ml.levels[0].P.tocsr()
    Kc = (P0.T @ K @ P0).tocsr()
    Mc = (P0.T @ M @ P0).tocsr()
    luc = spla.splu((Kc + sigma * Mc).tocsc())
    mc = _moden_direkt(Kc, Mc, sigma, luc, 2.0 * kcut2)
    del luc, Kc, Mc
    nm, npp = mc.shape[1], Pc.shape[1]
    if log:
        log('    AMG %s, coarse modes %d (%.1f s)' % ([lv.A.shape[0] for lv in ml.levels], nm, time.time() - t0))
    X = np.empty((n, 2 * nm + npp + nS))
    for j in range(0, nm, 16):
        X[:, j:min(j + 16, nm)] = P0 @ mc[:, j:j + 16]
    # (K+σM)⁻¹·[M·Moden, Anschlüsse]
    loese_viele(A, lambda j: M @ X[:, j] if j < nm else Pc[:, j - nm], X[:, nm:2 * nm + npp], ml, prozesse, log)
    if nS:
        o = 2 * nm
        loese_viele(A, lambda j: M @ X[:, o + j], X[:, 2 * nm + npp:], ml, prozesse, log)
    return X, nm


AMG_OPT = {'max_coarse': 2000, 'strength': ('symmetric', {'theta': 0.0})}


def _cg(A, b, prec, rtol=1e-7):
    x, info = spla.cg(A, b, M=prec, rtol=rtol, maxiter=200)
    return x


_WORKER = {}


def _worker_init(pfad, amg_opt):
    import pyamg
    d = np.load(pfad)
    A = sp.csr_matrix((d['data'], d['indices'], d['indptr']), shape=tuple(d['shape']))
    ml = pyamg.smoothed_aggregation_solver(A, B=np.ones((A.shape[0], 1)), **amg_opt)
    _WORKER['A'], _WORKER['prec'] = A, ml.aspreconditioner()


def _worker_solve(auftrag):
    ypfad, xpfad, j = auftrag
    Y = np.load(ypfad, mmap_mode='r')
    X = np.load(xpfad, mmap_mode='r+')
    X[:, j] = _cg(_WORKER['A'], np.array(Y[:, j]), _WORKER['prec'])
    X.flush()
    del X, Y
    return j


def _anzahl_prozesse(A, n, m, prozesse):
    import os
    import sys
    if prozesse is None:
        prozesse = max(1, min(6, (os.cpu_count() or 2) - 2))
        # Speicher: je Prozess etwa 60 Byte je Matrixeintrag
        prozesse = min(prozesse, max(1, int(4e9 // (60 * A.nnz + 1))))
    hauptdatei = getattr(sys.modules.get('__main__'), '__file__', None)
    if not (hauptdatei and os.path.isfile(hauptdatei)):
        prozesse = 1   # Prozesse brauchen ein importierbares Hauptprogramm (nicht bei stdin/interaktiv)
    if m < 2 * prozesse or n < 50000:
        prozesse = 1
    return prozesse


def loese_viele(A, spalte, out, ml, prozesse=None, log=None):
    """out[:, j] = (K+σM)⁻¹·spalte(j) für alle Spalten von out; große Netze parallel in Prozessen
    (rechte Seiten und Lösungen laufen über Dateien, damit der Hauptprozess keine Kopien hält)."""
    import os
    t0 = time.time()
    n, m = out.shape
    # Mit NVIDIA-GPU (CuPy) alle Spalten blockweise dort; geht dabei etwas schief, rechnet der Prozessor wie bisher
    if n >= 50000 and m >= 2:
        import gpu
        if gpu.verfuegbar()[0]:
            try:
                return gpu.loese_viele(A, spalte, out, ml, log)
            except Exception as e:  # noqa: BLE001 – z. B. Grafikspeicher zu klein
                gpu.abschalten('error: %s' % e)
                if log:
                    log('    GPU not usable (%s), calculating on the CPU' % e)
    prozesse = _anzahl_prozesse(A, n, m, prozesse)
    fertig = False
    if prozesse > 1:
        import tempfile
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        tmp = tempfile.mkdtemp(prefix='rs3d_')
        apfad, ypfad, xpfad = (os.path.join(tmp, x) for x in ('A.npz', 'Y.npy', 'X.npy'))
        try:
            np.savez(apfad, data=A.data, indices=A.indices, indptr=A.indptr, shape=np.array(A.shape))
            Y = np.lib.format.open_memmap(ypfad, mode='w+', shape=(n, m), fortran_order=True)
            for j in range(m):
                Y[:, j] = spalte(j)
            Y.flush()
            del Y
            Xm = np.lib.format.open_memmap(xpfad, mode='w+', shape=(n, m), fortran_order=True)
            del Xm
            with ProcessPoolExecutor(prozesse, initializer=_worker_init, initargs=(apfad, AMG_OPT)) as ex:
                list(ex.map(_worker_solve, [(ypfad, xpfad, j) for j in range(m)]))
            Xm = np.load(xpfad, mmap_mode='r')
            for j in range(m):
                out[:, j] = Xm[:, j]
            del Xm
            fertig = True
        except (BrokenProcessPool, OSError) as e:
            if log:
                log('    Parallel calculation not possible (%s), calculating serially' % e)
            prozesse = 1
        finally:
            for x in (apfad, ypfad, xpfad):
                try:
                    os.remove(x)
                except OSError:
                    pass
            try:
                os.rmdir(tmp)
            except OSError:
                pass
    if not fertig:
        prozesse = 1
        prec = ml.aspreconditioner()
        for j in range(m):
            out[:, j] = _cg(A, spalte(j), prec)
    if log:
        log('    %d solutions in %.1f s (%s)' % (m, time.time() - t0, '%d processes' % prozesse if prozesse > 1 else 'serial'))
    return out
