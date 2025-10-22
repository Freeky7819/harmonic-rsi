
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

@dataclass
class ISMConfig:
    window: int = 5
    downsample: int = 1
    center: str = "pca"
    scale: str = "l2"
    clip_quantile: float = 0.0
    use_log_time: bool = True
    use_timestamps: bool = False
    omega_hint: Optional[float] = None
    phi_hint: Optional[float] = None
    omega_min: float = 2.0
    omega_max: float = 10.0
    omega_steps: int = 64
    alpha: float = 0.08
    use_embeddings: bool = True
    use_loss: bool = False
    use_entropy: bool = False
    w_emb: float = 0.6
    w_loss: float = 0.2
    w_entropy: float = 0.2

class ISMField:
    def __init__(self, cfg: Optional[ISMConfig] = None):
        self.cfg = cfg or ISMConfig()
        self._n: int = 0
        self._u: Optional[np.ndarray] = None
        self._S: Optional[np.ndarray] = None
        self._Phi: Optional[np.ndarray] = None
        self._omega: Optional[float] = None
        self._phi: Optional[float] = None
        self._r2: Optional[float] = None

    def fit(self, trace: Dict[str, Any]) -> "ISMField":
        S, u = self._compose(trace)
        self._S, self._u = S, u
        self._n = len(S)
        omega, phi, r2 = self._estimate_phase(S, u, self.cfg)
        self._omega, self._phi, self._r2 = omega, phi, r2
        Phi = self._phi_series(S, u, omega, phi, self.cfg.alpha, self.cfg.window)
        self._Phi = Phi
        return self

    def signal(self, n_steps: Optional[int] = None) -> np.ndarray:
        assert self._Phi is not None, "Call fit(trace) first."
        k = self._Phi if n_steps is None else self._Phi[: int(n_steps)]
        return k.copy()

    def dphi(self, n_steps: Optional[int] = None) -> np.ndarray:
        Phi = self.signal(n_steps)
        if len(Phi) <= 1:
            return np.zeros_like(Phi)
        d = np.empty_like(Phi)
        d[1:-1] = 0.5 * (Phi[2:] - Phi[:-2])
        d[0] = Phi[1] - Phi[0]
        d[-1] = Phi[-1] - Phi[-2]
        return _smooth(d, max(1, min(5, self.cfg.window//2 or 1)))

    def estimate_phase(self) -> Dict[str, float]:
        assert self._omega is not None and self._phi is not None and self._r2 is not None, "fit(trace) first."
        return {"omega": float(self._omega), "phi": float(self._phi), "r2": float(self._r2)}

    def _compose(self, trace: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        emb = trace.get("embeddings", None)
        if emb is None:
            raise ValueError("embeddings required for ISM demo")
        X = np.asarray(emb, dtype=float)
        pc1 = _emb_to_pc1(X)
        S = _normalize_1d(pc1)
        n = len(S)
        if self.cfg.use_timestamps and trace.get("timestamps") is not None:
            t = np.asarray(trace["timestamps"][:n], dtype=float)
            t = t - float(t[0] if len(t) else 0.0) + 1e-9
            u = np.log(t) if self.cfg.use_log_time else t
        else:
            idx = np.arange(1, n + 1, dtype=float)
            u = np.log(idx) if self.cfg.use_log_time else idx
        return S, u

    @staticmethod
    def _estimate_phase(S: np.ndarray, u: np.ndarray, cfg: ISMConfig) -> Tuple[float, float, float]:
        omega_grid = np.array([cfg.omega_hint], dtype=float) if cfg.omega_hint is not None else np.linspace(cfg.omega_min, cfg.omega_max, cfg.omega_steps, dtype=float)
        best = (-np.inf, cfg.omega_min, 0.0, 0.0)
        for w in omega_grid:
            sinw = np.sin(w * u); cosw = np.cos(w * u)
            A = np.column_stack([np.ones_like(u), sinw, cosw])
            coef, _, _, _ = np.linalg.lstsq(A, S, rcond=None)
            yhat = A @ coef
            r2 = _r2_score(S, yhat)
            if r2 > best[0]:
                a, b, c = coef
                phi = -np.arctan2(c, b)
                best = (r2, float(w), float(phi), float(a))
        w0 = best[1]
        local = np.linspace(max(cfg.omega_min, w0 - 0.5), min(cfg.omega_max, w0 + 0.5), 21)
        for w in local:
            sinw = np.sin(w * u); cosw = np.cos(w * u)
            A = np.column_stack([np.ones_like(u), sinw, cosw])
            coef, _, _, _ = np.linalg.lstsq(A, S, rcond=None)
            yhat = A @ coef
            r2 = _r2_score(S, yhat)
            if r2 > best[0]:
                a, b, c = coef
                phi = -np.arctan2(c, b)
                best = (r2, float(w), float(phi), float(a))
        return float(best[1]), float(best[2]), float(best[0])

    @staticmethod
    def _phi_series(S: np.ndarray, u: np.ndarray, omega: float, phi: float, alpha: float, window: int) -> np.ndarray:
        mod = np.exp(-alpha * np.sin(omega * u + phi))
        Phi = S * mod
        return _smooth(Phi, window)

def _normalize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    mu, sig = np.mean(x), np.std(x) + 1e-9
    return (x - mu) / sig

def _emb_to_pc1(emb: np.ndarray) -> np.ndarray:
    X = np.asarray(emb, dtype=float)
    Xc = X - np.mean(X, axis=0, keepdims=True)
    nrm = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9
    Xc = Xc / nrm
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1_vec = Vt[0]
    pc1 = Xc @ pc1_vec
    return pc1

def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1 or len(x) < 3:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(window, dtype=float) / window
    y = np.convolve(xpad, ker, mode="same")[pad:-pad]
    return y

def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2)) + 1e-12
    return float(max(0.0, min(1.0, 1.0 - ss_res / ss_tot)))
