
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np
from .researcher import ResearcherAgent
from .base import Critic
from ..ism_field import ISMField, ISMConfig
from ..rsi import ResonanceEvaluator, ResonanceParams

@dataclass
class AdaptConfig:
    cycles: int = 3
    alpha_min: float = 0.04
    alpha_max: float = 0.16
    alpha_step: float = 0.02
    omega_min: float = 4.0
    omega_max: float = 9.5
    omega_steps: int = 6
    target: str = "maximize_rsig"

class AdaptiveResearcherAgent(ResearcherAgent):
    def score_tuple(self, res: dict, est: dict, target: str) -> Tuple[float, float, float]:
        rsig  = float(res.get("resonance_signature", 0.0))
        r2    = float(est.get("r2", 0.0))
        rsi   = float(res.get("resonance_score", 0.0))
        if target == "maximize_r2":  return (r2, rsig, rsi)
        if target == "maximize_rsi": return (rsi, rsig, r2)
        return (rsig, r2, rsi)

    def _evaluate_with(self, steps: List[str], emb: np.ndarray, alpha: float, omega: float) -> Tuple[dict, dict]:
        trace = {"steps": steps, "embeddings": emb}
        field = ISMField(ISMConfig(alpha=alpha, use_log_time=True, omega_hint=omega)).fit(trace)
        est = field.estimate_phase()
        ev  = ResonanceEvaluator()
        res = ev.evaluate(steps, mode="resonant", params=ResonanceParams(alpha=alpha, omega=omega, phi=est["phi"]))
        return res, est

    def run_adaptive(self, task: str, cfg: AdaptConfig = AdaptConfig()) -> Dict[str, Any]:
        base = super().run(task, alpha=cfg.alpha_min + (cfg.alpha_max - cfg.alpha_min)/2)
        steps = base["report"]["steps"]
        import numpy as _np
        emb = _np.asarray(base["trace"]["embeddings"], dtype="float32")

        trace = {"steps": steps, "embeddings": emb}
        field0 = ISMField(ISMConfig(alpha=0.08, use_log_time=True)).fit(trace)
        est0 = field0.estimate_phase()
        ev = ResonanceEvaluator()
        res0 = ev.evaluate(steps, mode="resonant", params=ResonanceParams(alpha=0.08, omega=est0["omega"], phi=est0["phi"]))
        best_res, best_est = res0, est0
        best_alpha, best_omega = 0.08, est0["omega"]
        best_score = self.score_tuple(best_res, best_est, cfg.target)

        alpha_vals = np.arange(cfg.alpha_min, cfg.alpha_max + 1e-9, cfg.alpha_step).tolist()
        omega_vals = np.linspace(cfg.omega_min, cfg.omega_max, cfg.omega_steps).tolist()

        def try_candidates(alphas, omegas):
            t_best = (best_alpha, best_omega, best_res, best_est, best_score)
            for a in alphas:
                for w in omegas:
                    res, est = self._evaluate_with(steps, emb, a, w)
                    score = self.score_tuple(res, est, cfg.target)
                    if score > t_best[4]:
                        t_best = (a, w, res, est, score)
            return t_best

        a, w, r, e, s = try_candidates(alpha_vals, omega_vals)
        if s > best_score:
            best_alpha, best_omega, best_res, best_est, best_score = a, w, r, e, s

        cur_alpha_step = cfg.alpha_step
        cur_omega_span = max(0.4, (cfg.omega_max - cfg.omega_min) / max(2, cfg.omega_steps-1))
        for _ in range(max(0, cfg.cycles - 1)):
            a_min = max(cfg.alpha_min, best_alpha - cur_alpha_step)
            a_max = min(cfg.alpha_max, best_alpha + cur_alpha_step)
            alphas = np.linspace(a_min, a_max, 3).tolist()

            w_min = max(cfg.omega_min, best_omega - cur_omega_span)
            w_max = min(cfg.omega_max, best_omega + cur_omega_span)
            omegas = np.linspace(w_min, w_max, max(3, cfg.omega_steps//2)).tolist()

            a, w, r, e, s = try_candidates(alphas, omegas)
            if s > best_score:
                best_alpha, best_omega, best_res, best_est, best_score = a, w, r, e, s

            cur_alpha_step = max(cfg.alpha_step/2, 0.01)
            cur_omega_span = max(cur_omega_span/2, 0.1)

        # Construct proper report structure matching baseline format
        report_data = {
            "steps": steps,
            "rsi": best_res,
            "ism_phase": best_est,
            "critic": base["report"].get("critic", ""),
            "final_text": base["report"].get("final_text", ""),
        }

        return {
            "baseline": {"rsi": res0, "ism_phase": est0, "alpha": 0.08, "omega": est0["omega"]},
            "adaptive": {"rsi": best_res, "ism_phase": best_est, "alpha": best_alpha, "omega": best_omega},
            "trace": base["trace"],
            "report": report_data,  # <-- ADDED: proper report key!
        }
