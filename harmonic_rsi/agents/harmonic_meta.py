
# harmonic_rsi/harmonic_meta.py
from __future__ import annotations
import os, json, time
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Dict, Any, List
from .researcher import ResearcherAgent
from .adaptive import AdaptiveResearcherAgent, AdaptConfig

from harmonic_rsi.agents.providers import OpenAIProvider, OllamaProvider
from harmonic_rsi.agents.researcher import ResearcherAgent
try:
    # optional; only if you have it
    from harmonic_rsi.agents.adaptive import AdaptiveResearcherAgent, AdaptConfig
    HAS_ADAPTIVE = True
except Exception:
    HAS_ADAPTIVE = False

@dataclass
class FilterConfig:
    # acceptance thresholds for the harmonic filter
    min_rsig: float = 0.62
    min_r2:   float = 0.60
    max_drift:float = 0.70
    min_coh:  float = 0.50

@dataclass
class MetaConfig:
    provider: Literal["openai","ollama"] = "openai"
    chat_model: str = "gpt-4o"
    emb_model:  str = "text-embedding-3-large"
    ollama_url: str = "http://localhost:11434"
    steps: int = 6
    use_adaptive: bool = False
    cycles: int = 1
    alpha: float = 0.08
    alpha_min: float = 0.06
    alpha_max: float = 0.10
    alpha_step: float = 0.02
    omega_min: float = 4.0
    omega_max: float = 9.5
    omega_steps: int = 4
    target: str = "maximize_rsig"
    # synthesis style
    final_from: Literal["baseline","adaptive"] = "baseline"

class MemoryStore:
    """Very small JSONL memory for session + persistent context."""
    def __init__(self, path: str = "memory.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8"): pass

    def add(self, item: Dict[str, Any]):
        item = dict(item)
        item["ts"] = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def last_k(self, k=10) -> List[Dict[str,Any]]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-k:]
            return [json.loads(x) for x in lines]
        except Exception:
            return []

class HarmonicMeta:
    """
    High-level orchestration with a harmonic filter + tiny memory.
    Usage:
        hm = HarmonicMeta()
        out = hm.ask("Zakaj je nebo modro", depth=1, reflect=True)
    """
    def __init__(self, meta: Optional[MetaConfig]=None, flt: Optional[FilterConfig]=None, memory_path="memory.jsonl"):
        self.cfg = meta or MetaConfig()
        self.flt = flt or FilterConfig()
        self.mem = MemoryStore(memory_path)
        # providers
        if self.cfg.provider == "openai":
            self.prov = OpenAIProvider(chat_model=self.cfg.chat_model, emb_model=self.cfg.emb_model)
        else:
            self.prov = OllamaProvider(base_url=self.cfg.ollama_url, chat_model=self.cfg.chat_model, emb_model=self.cfg.emb_model)

    def _run_baseline(self, task: str):
        agent = ResearcherAgent(self.prov, steps_n=self.cfg.steps, mode="resonant")
        return agent.run(task, alpha=self.cfg.alpha)

    def _run_adaptive(self, task: str):
        if not HAS_ADAPTIVE:
            # graceful fallback
            return self._run_baseline(task)
        cfg = AdaptConfig(
            cycles=self.cfg.cycles,
            alpha_min=self.cfg.alpha_min, alpha_max=self.cfg.alpha_max, alpha_step=self.cfg.alpha_step,
            omega_min=self.cfg.omega_min, omega_max=self.cfg.omega_max, omega_steps=self.cfg.omega_steps,
            target=self.cfg.target,
        )
        agent = AdaptiveResearcherAgent(self.prov, steps_n=self.cfg.steps, mode="resonant")
        return agent.run_adaptive(task, cfg=cfg)

    def _normalize_out(self, out):
        """
        Normalizira izhode agenta v obliko z 'report' in 'trace'.
        Sprejme:
          - {"report": {...}, "trace": {...}}
          - {"adaptive": {...}}                 (vzame adaptive blok)
          - "flat" slovar (npr. {"steps":..., "rsi":..., ...})  -> ovije v {"report": {...}}
          - karkoli nedict -> prazen skelet
        """
        if not isinstance(out, dict):
            return {"report": {}, "trace": {}}

        if "report" in out and isinstance(out["report"], dict):
            return out

        if "adaptive" in out and isinstance(out["adaptive"], dict):
            # nekateri adaptive agenti vrnejo 'adaptive' kot glavni blok
            return out["adaptive"]

        rep = {k: v for k, v in out.items() if k != "trace"}
        tr  = out.get("trace", {})
        return {"report": rep, "trace": tr}

    def _passes_filter(self, rsi: Dict[str, Any], est: Dict[str, Any]) -> bool:
        rsig = float(rsi.get("resonance_signature", 0.0))
        r2   = float(est.get("r2", 0.0))
        drift= float(rsi.get("phase_drift", 1.0))
        coh  = float(rsi.get("semantic_coherence", 0.0))
        return (rsig >= self.flt.min_rsig and r2 >= self.flt.min_r2 and drift <= self.flt.max_drift and coh >= self.flt.min_coh)

    def ask(self, task: str, depth: int = 1, reflect: bool = True, **_ignored) -> Dict[str, Any]:
        """
        **_ignored požre dodatne parametre iz UI (npr. use_memory, memory_text),
        da se izognemo 'unexpected keyword argument'.
        """
        history = self.mem.last_k(5)
        meta_note = f"Context memory (last {len(history)} items): " + "; ".join(
            h.get('summary', '') for h in history if h.get('summary')
        )

        best = None
        for round_id in range(max(1, depth)):
            # baseline
            base_raw = self._run_baseline(task)
            base     = self._normalize_out(base_raw)

            # adaptive (če vklopljeno)
            adapt = None
            if self.cfg.use_adaptive:
                adapt_raw = self._run_adaptive(task)
                adapt     = self._normalize_out(adapt_raw)

            # izberi vir
            source = adapt if (self.cfg.use_adaptive and self.cfg.final_from == "adaptive" and adapt) else base

            report = source.get("report", {})
            steps  = report.get("steps", []) or []
            rsi    = report.get("rsi",   {}) or {}
            est    = report.get("ism_phase", {}) or {}
            critic = report.get("critic", "") or ""
            final  = (report.get("final_text", "") or "").strip()

            # fallback, če agent ne vrne final_text
            if not final and steps:
                try:
                    sys = {"role":"system","content":"You are a clear, precise explainer. Answer in 5–8 sentences for a layperson."}
                    usr = {"role":"user","content": "Question: " + task + "\nPlan:\n- " + "\n- ".join(steps)}
                    final = (self.prov.chat([sys, usr]) or "").strip()
                except Exception:
                    final = "Summary:\n" + "\n".join(f"- {s}" for s in steps)

            accepted = self._passes_filter(rsi, est)

            cand = {
                "round": round_id + 1,
                "accepted": accepted,
                "source": "adaptive" if (self.cfg.use_adaptive and self.cfg.final_from=="adaptive" and adapt) else "baseline",
                "steps": steps, "critic": critic,
                "final_text": final,
                "rsi": rsi, "ism_phase": est,
                "trace": source.get("trace", {}),
            }
            if (best is None) or (float(cand["rsi"].get("resonance_signature", 0.0)) >
                                  float((best or {"rsi":{}})["rsi"].get("resonance_signature", 0.0))):
                best = cand

            # reflection, če filter pade
            if reflect and not accepted and final:
                sys = {"role":"system","content":"Rewrite the final answer more clearly and precisely for a layperson; keep it concise."}
                usr = {"role":"user","content": f"Task: {task}\n\nCurrent final answer:\n{final}\n\nNotes:\n{critic}\n\n{meta_note}"}
                try:
                    improved = (self.prov.chat([sys, usr]) or "").strip()
                    if improved:
                        cand["final_text"] = improved
                        best = cand
                except Exception:
                    pass

        try:
            self.mem.add({"summary": f"Task: {task}", "rsi": best["rsi"], "ism": best["ism_phase"]})
        except Exception:
            pass

        return {
            "ok": True,
            "config": asdict(self.cfg),
            "filter": asdict(self.flt),
            "result": best,
        }
