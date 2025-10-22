
from __future__ import annotations
import argparse, json, math, sys
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

_EMBED = None
_BACKEND = "fallback-hash-128"
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _EMBED = SentenceTransformer("all-MiniLM-L6-v2")
    _BACKEND = "sentence_transformers:all-MiniLM-L6-v2"
except Exception:
    _EMBED = None

@dataclass
class ResonanceParams:
    alpha: float = 0.08
    omega: Optional[float] = None
    phi: Optional[float] = None

class ResonanceEvaluator:
    def __init__(self, backend: str = "auto"):
        self.backend = _BACKEND if backend == "auto" else backend

    def evaluate(self, reasoning_trace: List[str], mode: str = "embedding", params: Optional[ResonanceParams] = None) -> dict:
        steps = [s.strip() for s in reasoning_trace if s and s.strip()]
        n = len(steps)
        if n == 0: return {"resonance_score":1.0,"phase_drift":0.0,"semantic_coherence":1.0,"resonance_signature":0.0,"backend":"n/a","mode":mode,"n_steps":0}
        if n == 1: return {"resonance_score":1.0,"phase_drift":0.0,"semantic_coherence":1.0,"resonance_signature":0.0,"backend":_BACKEND,"mode":mode,"n_steps":1}
        X = self._encode(steps)
        if mode == "embedding":
            drift = self._cos_drift(X); rsi, coh = self._scores(drift)
            return {"resonance_score": rsi, "phase_drift": float(np.mean(drift)), "semantic_coherence": coh, "resonance_signature": 0.0, "backend": self.backend, "mode": "embedding", "n_steps": n}
        rp = params or ResonanceParams()
        omega, phi = (rp.omega, rp.phi) if (rp.omega is not None and rp.phi is not None) else (6.0, 0.3)
        modX = self._apply_resonant(X, alpha=rp.alpha, omega=omega, phi=phi)
        drift = self._cos_drift(modX); rsi, coh = self._scores(drift)
        sig = self._signature(drift, omega, phi)
        return {"resonance_score": rsi, "phase_drift": float(np.mean(drift)), "semantic_coherence": coh, "resonance_signature": sig, "backend": self.backend, "mode": "resonant", "params":{"alpha":rp.alpha,"omega":omega,"phi":phi}, "n_steps": n}

    def _encode(self, steps: List[str]) -> np.ndarray:
        if _EMBED is not None and self.backend.startswith("sentence_transformers"):
            V = _EMBED.encode(steps, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(V, dtype=np.float32)
        rng = np.random.default_rng(123456789)
        basis = rng.standard_normal((128,256)).astype(np.float32)
        def s256(t: str) -> np.ndarray:
            b = t.encode("utf-8", errors="ignore")
            arr = np.zeros(256, dtype=np.float32)
            if b:
                for i in range(256): arr[i] = b[i % len(b)]
                arr /= (np.linalg.norm(arr)+1e-9)
            return arr
        raw = np.stack([s256(s) for s in steps], axis=0)
        vecs = raw @ basis.T
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True)+1e-9)
        self.backend = "fallback-hash-128"
        return vecs.astype(np.float32)

    @staticmethod
    def _cos_drift(X: np.ndarray) -> np.ndarray:
        V1, V2 = X[:-1], X[1:]
        V1 /= (np.linalg.norm(V1, axis=1, keepdims=True)+1e-9)
        V2 /= (np.linalg.norm(V2, axis=1, keepdims=True)+1e-9)
        cos = np.sum(V1*V2, axis=1).clip(-1.0,1.0)
        return 1.0 - cos

    @staticmethod
    def _scores(drift: np.ndarray) -> Tuple[float,float]:
        m = float(np.mean(drift))
        return round(np.exp(-m),3), round(max(0.0, min(1.0, 1.0-m)),3)

    @staticmethod
    def _apply_resonant(X: np.ndarray, alpha: float, omega: float, phi: float) -> np.ndarray:
        n = X.shape[0]; idx = np.arange(1,n+1,dtype=np.float32)
        mod = np.exp(-alpha*np.sin(omega*np.log(idx)+phi)); mod = np.clip(mod,0.5,1.5)
        return (X.T*mod).T.astype(np.float32)

    @staticmethod
    def _signature(drift: np.ndarray, omega: float, phi: float) -> float:
        m = drift.shape[0]
        if m < 3: return 0.0
        idx = np.arange(1,m+1,dtype=np.float32); phase = omega*np.log(idx)+phi
        S, C = np.sin(phase), np.cos(phase)
        A = np.column_stack([np.ones_like(S), S, C])
        coef, *_ = np.linalg.lstsq(A, drift, rcond=None); y = A @ coef
        ss_res = float(np.sum((drift - y)**2)); ss_tot = float(np.sum((drift - np.mean(drift))**2))+1e-12
        return float(max(0.0, min(1.0, 1.0 - ss_res/ss_tot)))

def cli_main(): return 0
if __name__=="__main__": import sys; sys.exit(cli_main())
