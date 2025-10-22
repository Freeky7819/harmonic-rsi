# ------------------------------------------------------------------------------
# Harmonic RSI — Research Edition (v0.2.0)
# Licensed under CC BY-NC 4.0 (non-commercial use only)
# © Damjan Žakelj 2025 — Harmonic Logos / ISM-X Framework
# ------------------------------------------------------------------------------


import json
import numpy as np
from .providers import OpenAIProvider, OllamaProvider
from ..ism_field import ISMField, ISMConfig
from ..rsi import ResonanceEvaluator, ResonanceParams

def _make_critic(provider, task: str, steps: list[str]) -> str:
    """Generate 2â€“3 short critic notes about the plan (no CoT)."""
    sys = {
        "role": "system",
        "content": ("You are a critical reviewer. Return 2-3 short bullet notes pointing out "
                    "missing specifics, verification steps, or possible pitfalls. "
                    "No chain-of-thought; only the final bullet notes.")
    }
    joined = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    usr = {"role": "user", "content": f"Task:\n{task}\n\nPlan steps:\n{joined}"}
    try:
        txt = provider.chat([sys, usr]).strip()
        return txt if txt else ""
    except Exception:
        return ""

def _synthesize_final(provider, task: str, steps: list[str]) -> str:
    """
    Produce a real, self-contained final answer for the user (not a plan).
    Language: match the user's task language if possible.
    """
    sys = {
        "role": "system",
        "content": ("Write the final answer for the user's task. "
                    "Be direct and self-contained. No numbered steps, no chain-of-thought. "
                    "Use the same language as the task. Keep it concise but complete.")
    }
    joined = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    usr = {"role": "user", "content": f"Task:\n{task}\n\nPlanned steps:\n{joined}\n\nNow provide the final answer (not a plan):"}
    try:
        txt = provider.chat([sys, usr]).strip()
        return txt
    except Exception:
        return ""

class ResearcherAgent:
    def __init__(self, provider, steps_n=6, mode="resonant"):
        self.provider = provider
        self.steps_n = steps_n
        self.mode = mode

    def run(self, task: str, alpha: float = 0.08):
        """Run the base Researcher agent and return full report with embeddings + final answer."""
        steps = self.provider.get_steps(task, self.steps_n)
        emb = self.provider.get_embeddings(steps)

        trace = {"steps": steps, "embeddings": emb.tolist()}
        cfg = ISMConfig(alpha=alpha, use_log_time=True)
        field = ISMField(cfg).fit(trace)
        est = field.estimate_phase()
        ev = ResonanceEvaluator()
        res = ev.evaluate(
            steps, mode=self.mode,
            params=ResonanceParams(alpha=cfg.alpha, omega=est["omega"], phi=est["phi"]),
        )

        # critic + final answer
        critic = _make_critic(self.provider, task, steps)
        final_text = _synthesize_final(self.provider, task, steps)
        if not final_text:
            # robust fallback so UI always shows something
            final_text = "\n".join(steps[-2:]) if len(steps) >= 2 else (steps[-1] if steps else "")

        report = {"steps": steps, "rsi": res, "ism_phase": est, "critic": critic, "final_text": final_text}

        return {
            "trace": trace,
            "report": report,
            "trace_json": json.dumps(trace, ensure_ascii=False, indent=2),
            "report_json": json.dumps(report, ensure_ascii=False, indent=2)
        }
