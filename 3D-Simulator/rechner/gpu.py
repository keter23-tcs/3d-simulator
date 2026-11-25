"""Lösen auf der Grafikkarte (NVIDIA, über CuPy). Freiwillig: ohne CuPy oder GPU rechnet fem.py wie bisher auf dem Prozessor.

Gebraucht wird es für den teuersten Schritt der 3D-Rechnung: dieselbe symmetrisch positiv definite Matrix A = K + σM
für einige hundert rechte Seiten lösen (fem.loese_viele). Auf der GPU laufen viele Spalten gleichzeitig:
  - CG je Spalte, aber alle Spalten eines Blocks im Gleichschritt (A·P als dünnbesetzt × Block, eine cuSPARSE-Operation),
  - Vorkonditionierer: die AMG-Stufen aus pyamg (P, R, A je Stufe) als V-Zyklus auf der GPU. Statt Gauß-Seidel (läuft
    nacheinander, passt nicht auf die GPU) glättet ein Tschebyschow-Polynom 2. Grades mit Diagonalskalierung; vor- und
    nachher dasselbe Polynom, damit der Vorkonditionierer symmetrisch bleibt (Voraussetzung für CG). Grobste Stufe: Inverse.
    Alles in float64: Ein Vorkonditionierer in float32 (auch nur auf der feinsten Stufe) ist nicht mehr exakt symmetrisch,
    beim 12pi brauchte CG damit 3–4 Mal so viele Schritte und war insgesamt langsamer (ausprobiert 23.10.2026).
  - Abbruch wie auf dem Prozessor: ‖r‖ ≤ rtol·‖b‖ je Spalte, höchstens maxiter Schritte.

Ausschalten: Umgebungsvariable RS3D_GPU=0 oder rs3d.py --cpu.
"""
import os
import time

import numpy as np

_STATUS = None


def verfuegbar():
    """(True, Name der GPU) oder (False, Grund). Wird einmal geprüft."""
    global _STATUS
    if _STATUS is None:
        if os.environ.get('RS3D_GPU', '1') == '0':
            _STATUS = (False, 'switched off (RS3D_GPU=0)')
        else:
            try:
                import warnings
                with warnings.catch_warnings():
                    # „CUDA path could not be detected“: die CUDA-Bibliotheken kommen aus dem pip-Paket, das passt
                    warnings.simplefilter('ignore', UserWarning)
                    import cupy as cp
                n = cp.cuda.runtime.getDeviceCount()
                if n < 1:
                    raise RuntimeError('no NVIDIA GPU found')
                (cp.arange(4.0) @ cp.arange(4.0)).item()  # lädt die CUDA-Bibliotheken einmal richtig
                name = cp.cuda.runtime.getDeviceProperties(0)['name']
                _STATUS = (True, name.decode() if isinstance(name, bytes) else str(name))
            except Exception as e:  # noqa: BLE001 – jeder Fehler heißt: Prozessor nehmen
                _STATUS = (False, '%s: %s' % (type(e).__name__, e))
    return _STATUS


def abschalten(grund='switched off'):
    global _STATUS
    _STATUS = (False, grund)


class _Stufe:
    def __init__(self, cp, csr, A, P=None, R=None, grad=2):
        self.grad = grad
        self.A = csr(A.tocsr())
        self.P = csr(P.tocsr()) if P is not None else None
        self.R = csr(R.tocsr()) if R is not None else None
        d = A.diagonal()
        self.dinv = cp.asarray(1.0 / d)[:, None]
        # größter Eigenwert von D⁻¹A per Potenzmethode; das Glättungsintervall ist [λmax/30, 1,1·λmax]
        x = cp.asarray(np.random.default_rng(1).standard_normal(A.shape[0]))[:, None]
        lam = 1.0
        for _ in range(15):
            y = self.dinv * (self.A @ x)
            lam = float(cp.linalg.norm(y) / cp.linalg.norm(x))
            x = y / cp.linalg.norm(y)
        hi = 1.1 * lam
        lo = hi / 30.0
        self.theta, self.delta = (hi + lo) / 2, (hi - lo) / 2

    def glaetten(self, B, X):
        """Tschebyschow-Glättung vom Grad self.grad; X=None heißt Startwert 0."""
        Rr = B.copy() if X is None else B - self.A @ X
        sigma = self.theta / self.delta
        rho = 1.0 / sigma
        D = (self.dinv * Rr) / self.theta
        X = D if X is None else X + D
        for k in range(1, self.grad):
            Rr -= self.A @ D
            rho_neu = 1.0 / (2 * sigma - rho)
            D = (rho_neu * rho) * D + (2 * rho_neu / self.delta) * (self.dinv * Rr)
            X += D
            rho = rho_neu
        return X


class Loeser:
    """GPU-Fassung von A⁻¹ für viele rechte Seiten; ml = pyamg-Hierarchie (smoothed aggregation) zu A."""

    def __init__(self, A, ml, grad=None):
        """grad = Grad des Tschebyschow-Glätters (Standard aus RS3D_GLAETTER, sonst 2)."""
        grad = int(grad or os.environ.get('RS3D_GLAETTER', 2))
        import cupy as cp
        import cupyx.scipy.sparse as csp
        self.cp = cp
        self.n = A.shape[0]
        self.stufen = []
        lv = ml.levels
        for i in range(len(lv) - 1):
            self.stufen.append(_Stufe(cp, csp.csr_matrix, lv[i].A, lv[i].P, lv[i].R, grad=grad))
        self.grob = cp.asarray(np.linalg.inv(lv[-1].A.toarray()))
        self.A = self.stufen[0].A if self.stufen else csp.csr_matrix(A.tocsr())

    def vork(self, Rr):
        """Vorkonditionierer: ein V-Zyklus."""
        return self._v(0, Rr)

    def _v(self, i, B):
        if i == len(self.stufen):
            return self.grob @ B
        s = self.stufen[i]
        X = s.glaetten(B, None)
        Xc = self._v(i + 1, s.R @ (B - s.A @ X))
        X += s.P @ Xc
        return s.glaetten(B, X)

    def block(self, B, rtol=1e-7, maxiter=200):
        """Löst A·X = B (B als n × b auf der GPU) mit vorkonditioniertem CG, alle Spalten im Gleichschritt."""
        cp = self.cp
        X = cp.zeros_like(B)
        Rr = B.copy()
        bn = cp.linalg.norm(B, axis=0)
        bn[bn == 0] = 1.0
        Z = self.vork(Rr)
        P = Z.copy()
        rz = cp.sum(Rr * Z, axis=0)
        aktiv = cp.ones(B.shape[1], dtype=bool)
        for it in range(maxiter):
            AP = self.A @ P
            alpha = cp.where(aktiv, rz / cp.sum(P * AP, axis=0), 0.0)
            X += P * alpha
            Rr -= AP * alpha
            aktiv &= cp.linalg.norm(Rr, axis=0) > rtol * bn
            if not bool(aktiv.any()):
                break
            Z = self.vork(Rr)
            rz_neu = cp.sum(Rr * Z, axis=0)
            beta = cp.where(aktiv, rz_neu / rz, 0.0)
            P = Z + P * beta
            rz = rz_neu
        return X, it + 1, int(aktiv.sum())

    def blockgroesse(self, m):
        """Wie viele Spalten gleichzeitig: etwa 60 % des freien Grafikspeichers; je Spalte 7 Vektoren float64 fürs CG
        und rund 6 in der Vorkonditionierer-Genauigkeit (feinste Stufe, die gröberen sind klein)."""
        frei = self.cp.cuda.Device().mem_info[0]
        je_spalte = self.n * 8 * 13 * 1.15
        return int(max(1, min(m, 128, 0.6 * frei // je_spalte)))


def loese_viele(A, spalte, out, ml, log=None, rtol=1e-7):
    """Wie fem.loese_viele, aber auf der GPU: out[:, j] = A⁻¹·spalte(j)."""
    import cupy as cp
    t0 = time.time()
    n, m = out.shape
    L = Loeser(A, ml)
    b = L.blockgroesse(m)
    schritte, offen = [], 0
    for j0 in range(0, m, b):
        j1 = min(m, j0 + b)
        B = np.empty((n, j1 - j0))
        for j in range(j0, j1):
            B[:, j - j0] = spalte(j)
        X, it, nicht = L.block(cp.asarray(B), rtol=rtol)
        offen += nicht
        out[:, j0:j1] = cp.asnumpy(X)
        schritte.append(it)
        del X
    if log:
        log('    %d solutions in %.1f s (GPU %s, blocks of %d columns, %s CG steps)'
            % (m, time.time() - t0, verfuegbar()[1], b, '/'.join(map(str, schritte))))
        if offen:
            log('    Warning: %d solutions did not reach the accuracy %.0e after 200 steps' % (offen, rtol))
    del L
    cp.get_default_memory_pool().free_all_blocks()
    return out
