# ------------------------------------------------------------------------------
# Harmonic RSI — Research Edition (v0.2.0)
# Licensed under CC BY-NC 4.0 (non-commercial use only)
# © Damjan Žakelj 2025 — Harmonic Logos / ISM-X Framework
# ------------------------------------------------------------------------------

# -*- coding: utf-8 -*-
# harmonic_rsi/app_gradio.py
from __future__ import annotations
import os, json, tempfile
from typing import List, Tuple

import gradio as gr
import numpy as np
import pandas as pd

from harmonic_rsi.agents.harmonic_meta import HarmonicMeta, MetaConfig, FilterConfig
from harmonic_rsi import ISMField, ISMConfig, ResonanceEvaluator, ResonanceParams

# ---- helpers -----------------------------------------------------------------

def _synth_critic(res: dict, est: dict, steps: list[str]) -> str:
    """Če agent ne vrne kritike, jo naredimo iz metrik in korakov."""
    rsi   = float(res.get("resonance_score", 0.0))
    rsig  = float(res.get("resonance_signature", 0.0))
    drift = float(res.get("phase_drift", 1.0))
    coh   = float(res.get("semantic_coherence", 0.0))
    r2    = float(est.get("r2", 0.0))

    notes = []
    notes.append(f"*RSIG={rsig:.3f}, RÂ˛={r2:.3f}, RSI={rsi:.3f}, DRIFT={drift:.3f}, COH={coh:.3f}*")
    if rsig < 0.60: notes.append("- Povečaj semantično konsistenco med koraki (prehodni stavki, manj topic skokov).")
    if r2   < 0.60: notes.append("- Fazna ocena ni stabilna -> poskusi manj korakov ali drugačen red.")
    if rsi  < 0.55: notes.append("- Koraki so premalo resonantni -> bolj konkretne, manj â€śmehkeâ€ť akcije.")
    if drift> 0.70: notes.append("- Prevelik DRIFT -> koraki naj bodo bolj medsebojno odvisni.")
    if coh  < 0.50: notes.append("- COH nizka -> dodaj povzetke med koraki, da drĹľiĹˇ rdečo nit.")
    if steps:
        notes.append("- Kritika korakov:")
        for i, s in enumerate(steps[:6], 1):
            notes.append(f"  {i}. {s}")
    return "\n".join(notes) if notes else "No critique available."
    

def _synth_final(steps: list[str]) -> str:
    """Če agent ne vrne finalnega odgovora, naredimo kratek â€śexecutive summaryâ€ť iz korakov."""
    if not steps:
        return "No final answer."
    bullets = "\n".join(f"- {s}" for s in steps[:10])
    return f"**Summary (auto):**\n{bullets}\n\n*(Auto-synth zaradi manjkajočega final_text v agentovem izhodu.)*"


def _badge_html(label: str, color: str, res: dict, est: dict) -> str:
    rsig = float(res.get("resonance_signature", 0.0))
    rsi  = float(res.get("resonance_score", 0.0))
    drift= float(res.get("phase_drift", 1.0))
    coh  = float(res.get("semantic_coherence", 0.0))
    r2   = float(est.get("r2", 0.0))
    pill = (
        f'<div style="display:inline-block;padding:6px 12px;'
        f'border-radius:9999px;background:{color};color:#111;'
        f'font-weight:700">{label}</div>'
    )
    nums = (
        f'<div style="margin-top:6px;font-family:monospace">'
        f'RSIG={rsig:.3f} · R²={r2:.3f} · RSI={rsi:.3f} · DRIFT={drift:.3f} · COH={coh:.3f}'
        f'</div>'
    )
    return pill + nums

def classify_stability(res: dict, est: dict) -> str:
    """3-state klasifikacija: Resonant / Stable / Chaotic drift."""
    rsig = float(res.get("resonance_signature", 0.0))
    rsi  = float(res.get("resonance_score", 0.0))
    drift= float(res.get("phase_drift", 1.0))
    coh  = float(res.get("semantic_coherence", 0.0))
    r2   = float(est.get("r2", 0.0))

    if rsig >= 0.60 and r2 >= 0.60 and rsi >= 0.55 and drift <= 0.70:
        return _badge_html("Resonant", "#22c55e", res, est)
    if coh < 0.40 or drift > 0.80:
        return _badge_html("Chaotic drift", "#ef4444", res, est)
    return _badge_html("Stable (non-resonant)", "#f59e0b", res, est)

def _eval_trace(steps: list[str], emb: np.ndarray, mode: str = "resonant"):
    """
    Skupna pot: ISM fit -> ocena faze -> RSI -> DataFrame za grafa.
    steps: seznam korakov (N)
    emb:   matrika embeddingov oblike (N, D)
    """
    # 1) zgradi minimalni trace, ki ga ISMField zna prebrati
    trace = {
        "steps": steps,
        "final_text": "",                   # v Evaluate nimamo končnega odgovora
        "embeddings": emb.tolist(),         # numpy -> list (json-serializable)
    }

    # 2) ISM fit + ocena faze
    cfg = ISMConfig(alpha=0.08, use_log_time=True)
    field = ISMField(cfg).fit(trace)
    est = field.estimate_phase()           # dict z "omega", "phi", "r2"
    Phi = field.signal()                   # Φ(t)
    dPhi = field.dphi()                    # ∂Φ/∂t

    # 3) RSI ocena
    ev = ResonanceEvaluator()
    res = ev.evaluate(
        steps or [f"step {i+1}" for i in range(len(Phi))],
        mode=mode,
        params=ResonanceParams(alpha=cfg.alpha, omega=est["omega"], phi=est["phi"]),
    )

    # 4) podatki za grafa
    x = list(range(1, len(Phi) + 1))
    df_phi  = pd.DataFrame({"x": x, "Φ(t)":  [float(v) for v in Phi]})
    df_dphi = pd.DataFrame({"x": x, "∂Φ/∂t": [float(v) for v in dPhi]})

    return res, est, df_phi, df_dphi, trace


def _tmp_write(text: str, fname: str) -> str|None:
    if not text:
        return None
    p = os.path.join(tempfile.gettempdir(), fname)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p

# ---- Evaluate tab (Prompt Ă˘â€ â€™ GPT Ă˘â€ â€™ Embeddings Ă˘â€ â€™ ISM Ă˘â€ â€™ RSI) --------------------

def _openai_steps(api_key: str, chat_model: str, steps_n: int, prompt: str) -> list[str]:
    """Vrne numbered korake (brez verige razmišljanja)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    text = ""
    try:
        # New Responses API
        resp = client.responses.create(
            model=chat_model,
            input=[
                {"role":"system","content":f"Respond with a numbered list of {steps_n} concrete actions. No chain-of-thought; only final steps."},
                {"role":"user","content": prompt},
            ],
        )
        text = resp.output[0].content[0].text
    except Exception:
        # Fallback Chat Completions API
        cc = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role":"system","content":f"Respond with a numbered list of {steps_n} concrete actions. No chain-of-thought; only final steps."},
                {"role":"user","content": prompt},
            ],
        )
        text = cc.choices[0].message.content or ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned: list[str] = []
    for ln in lines:
        # odstrani bullets/ÄąË‡tevilke
        ln = ln.lstrip("-* ").lstrip("0123456789").lstrip(".)").strip()
        if ln:
            cleaned.append(ln)
    return cleaned[:steps_n] or ([text.strip()] if text.strip() else [])

def _openai_embeddings(api_key: str, emb_model: str, steps: list[str]) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    inputs = [f"[step {i+1}] {s}" for i, s in enumerate(steps)]
    em = client.embeddings.create(model=emb_model, input=inputs)
    return np.array([e.embedding for e in em.data], dtype="float32")

def run_from_prompt(api_key, chat_model, emb_model, steps_n, user_prompt, mode):
    try:
        key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key:
            raise ValueError("Provide API key or set OPENAI_API_KEY.")

        # 1. Pridobi korake iz LLM
        steps = _openai_steps(key, chat_model, int(steps_n), user_prompt)
        if not steps:
            raise RuntimeError("LLM returned empty steps.")

        # 2. Ustvari embeddinge
        emb = _openai_embeddings(key, emb_model, steps)

        # 3. Oceni ISM/RSI
        res, est, df_phi, df_dphi, trace_obj = _eval_trace(steps, emb, mode)

        # 4. Serializacija in download poti
        trace_json  = json.dumps(trace_obj, ensure_ascii=False, indent=2)
        report_json = json.dumps(
            {"steps": steps, "final_text": "", "rsi": res, "ism_phase": est},
            ensure_ascii=False, indent=2
        )
        trace_path  = _tmp_write(trace_json,  "trace.json")
        report_path = _tmp_write(report_json, "report.json")

        # 5. Diagnostični izpisi
        diag_html = classify_stability(res, est)
        steps_txt = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

        return (
            diag_html,
            steps_txt,
            json.dumps(res, indent=2),
            json.dumps(est, indent=2),
            df_phi,
            df_dphi,
            trace_path,
            report_path,
        )

    except Exception as e:
        err = f"Error: {type(e).__name__}: {e}"
        return err, err, err, err, None, None, None, None

def run_from_file(file_obj, mode):
    try:
        if file_obj is None:
            raise ValueError("Upload a JSON trace (steps + embeddings).")
        path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        data = json.load(open(path, "r", encoding="utf-8"))
        steps = data.get("steps") or []
        emb = np.array(data.get("embeddings"), dtype="float32")
        res, est, df_phi, df_dphi = _eval_trace(steps, emb, mode)
        diag_html = classify_stability(res, est)
        steps_txt = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        return diag_html, steps_txt, json.dumps(res, indent=2), json.dumps(est, indent=2), df_phi, df_dphi
    except Exception as e:
        err = f"Error: {type(e).__name__}: {e}"
        return err, err, err, err, None, None

# ---- Meta-Agent tab (OpenAI/Ollama, z Adaptive mode) -------------------------

def run_meta_agent(
    provider, chat_model, emb_model, ollama_url,
    task, steps_n, mode, alpha,
    adaptive, cycles, alpha_min, alpha_max, alpha_step,
    omega_min, omega_max, omega_steps, target
):
    try:
        from harmonic_rsi.agents.providers import OpenAIProvider, OllamaProvider
        from harmonic_rsi.agents.researcher import ResearcherAgent
        from harmonic_rsi.agents.adaptive import AdaptiveResearcherAgent, AdaptConfig

        # 1) provider
        prov = (
            OpenAIProvider(chat_model=chat_model, emb_model=emb_model)
            if provider == "openai"
            else OllamaProvider(base_url=ollama_url, chat_model=chat_model, emb_model=emb_model)
        )

        # 2) poĹľeni agent(a)
        if adaptive:
            agent = AdaptiveResearcherAgent(prov, steps_n=int(steps_n), mode=mode)
            cfg = AdaptConfig(
                cycles=int(cycles),
                alpha_min=float(alpha_min), alpha_max=float(alpha_max), alpha_step=float(alpha_step),
                omega_min=float(omega_min), omega_max=float(omega_max), omega_steps=int(omega_steps),
                target=target,
            )
            out = agent.run_adaptive(task, cfg=cfg)
            # ---- ADAPTIVE NORMALIZACIJA ----
            steps     = out.get("steps", [])
            block     = out.get("adaptive", {})          # <-- rsi/phase sta tu
            res       = block.get("rsi", {})
            est       = block.get("ism_phase", {})
            critic    = out.get("critic", "")            # <-- pri adaptivu je kritik na top-level
            final_txt = ""                               # adaptive običajno ne vrne final_text
            trace_obj = out.get("trace", {})
        else:
            agent = ResearcherAgent(prov, steps_n=int(steps_n), mode=mode)
            out   = agent.run(task, alpha=float(alpha))
            # ---- NON-ADAPTIVE NORMALIZACIJA ----
            report   = out.get("report", {})
            steps    = report.get("steps", [])
            res      = report.get("rsi", {})
            est      = report.get("ism_phase", {})
            critic   = report.get("critic", "")
            final_txt= report.get("final_text", "")
            trace_obj= out.get("trace", {})

        # 3) grafi (če je trace)
        df_phi = df_dphi = None
        if trace_obj:
            field = ISMField(ISMConfig(alpha=float(alpha), use_log_time=True)).fit(trace_obj)
            Phi  = field.signal(); dPhi = field.dphi()
            x = list(range(1, len(Phi)+1))
            df_phi  = pd.DataFrame({"x": x, "Φ(t)":  [float(v) for v in Phi]})
            df_dphi = pd.DataFrame({"x": x, "∂Φ/∂t": [float(v) for v in dPhi]})

        # 4) izvoz
        trace_path  = _tmp_write(json.dumps(trace_obj or {}, ensure_ascii=False, indent=2), "agent_trace.json")
        report_path = _tmp_write(json.dumps(out or {},        ensure_ascii=False, indent=2), "agent_report.json")

        # 5) UI izpisi (+ robusten fallback za final)
        diag_html = classify_stability(res or {}, est or {})
        steps_txt = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)) if steps else ""
        if not final_txt:  # fallback = kratek povzetek korakov
            final_txt = "\n".join(f"- {s}" for s in steps) if steps else "No answer available."

        return (
            diag_html, steps_txt, (critic or ""),
            json.dumps(res or {}, indent=2),
            json.dumps(est or {}, indent=2),
            df_phi, df_dphi, trace_path, report_path,
            final_txt
        )
    except Exception as e:
        err = f"Error: {type(e).__name__}: {e}"
        return err, err, err, err, err, None, None, None, None, "No final answer."

def run_meta_ask(
    provider, chat_model, emb_model, ollama_url,
    task, steps_n, mode, alpha,
    adaptive, cycles, alpha_min, alpha_max, alpha_step,
    omega_min, omega_max, omega_steps, target,
    memory_text, use_memory
):
    try:
        meta = MetaConfig(
            provider=provider, chat_model=chat_model, emb_model=emb_model, ollama_url=ollama_url,
            steps=int(steps_n), use_adaptive=bool(adaptive), cycles=int(cycles),
            alpha=float(alpha), alpha_min=float(alpha_min), alpha_max=float(alpha_max),
            alpha_step=float(alpha_step), omega_min=float(omega_min), omega_max=float(omega_max),
            omega_steps=int(omega_steps), target=target,
            final_from=("adaptive" if adaptive else "baseline"),
        )
        flt = FilterConfig(min_rsig=0.62, min_r2=0.60, max_drift=0.70, min_coh=0.50)

        hm  = HarmonicMeta(meta=meta, flt=flt, memory_path="memory.jsonl")
        out = hm.ask(task, depth=1, reflect=True, use_memory=bool(use_memory))
        res = out["result"]  # <â€” harmoniziran rezultat

        steps     = res.get("steps", [])
        critic    = res.get("critic", "")
        final_txt = res.get("final_text", "")
        rsi       = res.get("rsi", {})
        est       = res.get("ism_phase", {})
        trace     = res.get("trace", {})

        # grafi
        df_phi = df_dphi = None
        if trace:
            field = ISMField(ISMConfig(alpha=float(alpha), use_log_time=True)).fit(trace)
            Phi = field.signal(); dPhi = field.dphi()
            x = list(range(1, len(Phi)+1))
            df_phi  = pd.DataFrame({"x": x, "Φ(t)":  [float(v) for v in Phi]})
            df_dphi = pd.DataFrame({"x": x, "∂Φ/∂t": [float(v) for v in dPhi]})

        trace_path  = _tmp_write(json.dumps(trace, ensure_ascii=False, indent=2), "meta_trace.json")
        report_path = _tmp_write(json.dumps(out,   ensure_ascii=False, indent=2), "meta_report.json")

        diag_html = classify_stability(rsi or {}, est or {})
        steps_txt = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        if not final_txt:
            bullets = "\n".join([f"- {s}" for s in steps[:8]]) if steps else ""
            final_txt = f"Summary (auto):\n{bullets}".strip()

        mem_text = out.get("memory_text", memory_text or "")
        return (
            diag_html, steps_txt, critic,
            json.dumps(rsi or {}, indent=2), json.dumps(est or {}, indent=2),
            df_phi, df_dphi, trace_path, report_path, final_txt, mem_text
        )
    except Exception as e:
        err = f"Error: {type(e).__name__}: {e}"
        return err, err, err, err, err, None, None, None, None, "No final answer.", (memory_text or "")


# ---- UI ----------------------------------------------------------------------

def main():
    with gr.Blocks(title="Harmonic RSI Ă˘â‚¬â€ť All-in-one (with Meta-Agent)") as demo:
        gr.Markdown("# Harmonic RSI Ă˘â‚¬â€ť All-in-one (with Meta-Agent)")

        # ===================== TAB 1: Evaluate =====================
        with gr.Tab("Evaluate"):
            with gr.Row():
                api_key    = gr.Textbox(label="OpenAI API Key (optional if set in env)", type="password", placeholder="sk-...")
                chat_model = gr.Dropdown(choices=["gpt-4o","gpt-4o-mini","gpt-4.1"], value="gpt-4o", label="Chat model")
                emb_model  = gr.Dropdown(choices=["text-embedding-3-large","text-embedding-3-small"], value="text-embedding-3-large", label="Embedding model")
                steps_n    = gr.Slider(3, 10, value=6, step=1, label="Number of steps")
                mode_eval  = gr.Radio(choices=["embedding","resonant"], value="resonant", label="RSI mode")

            user_prompt = gr.Textbox(lines=4, label="Prompt", placeholder="Describe the task to plan (agent workflow)...")
            run_btn     = gr.Button("Generate & Evaluate")
            file        = gr.File(file_types=[".json"], label="Upload trace JSON")
            run_file_btn= gr.Button("Evaluate Uploaded Trace")

            diag_out = gr.HTML()
            with gr.Row():
                steps_out = gr.Code(label="Steps")
            with gr.Row():
                rsi_out = gr.Code(label="RSI result")
                ism_out = gr.Code(label="ISM phase (ÄŽâ€°, ÄŽâ€ , R²)")
            with gr.Row():
                plot_phi  = gr.LinePlot(x="x", y="ĂŽÂ¦(t)",  label="ĂŽÂ¦(t)")
                plot_dphi = gr.LinePlot(x="x", y="Ă˘Ââ€šĂŽÂ¦/Ă˘Ââ€št", label="Ă˘Ââ€šĂŽÂ¦/Ă˘Ââ€št")
            trace_dl  = gr.File(label="Download trace.json (generated)")
            report_dl = gr.File(label="Download report.json")

            run_btn.click(
                run_from_prompt,
                inputs=[api_key, chat_model, emb_model, steps_n, user_prompt, mode_eval],
                outputs=[diag_out, steps_out, rsi_out, ism_out, plot_phi, plot_dphi, trace_dl, report_dl]
            )

            run_file_btn.click(
                run_from_file,
                inputs=[file, mode_eval],
                outputs=[diag_out, steps_out, rsi_out, ism_out, plot_phi, plot_dphi]
            )

        # ===================== TAB 2: Meta-Agent =====================
        with gr.Tab("Meta-Agent"):
            with gr.Row():
                provider   = gr.Radio(choices=["openai","ollama"], value="openai", label="Provider")
                chat_m     = gr.Textbox(value="gpt-4o", label="Chat model / Ollama model")
                emb_m      = gr.Textbox(value="text-embedding-3-large", label="Embedding model / Ollama embed model")
                ollama_url = gr.Textbox(value="http://localhost:11434", label="Ollama URL")
                steps2     = gr.Slider(3, 10, value=6, step=1, label="Steps")
                mode2      = gr.Radio(choices=["embedding","resonant"], value="resonant", label="RSI mode")
                alpha2     = gr.Slider(0.02, 0.20, value=0.08, step=0.01, label="alpha (baseline)")

            with gr.Row():
                adaptive_cb  = gr.Checkbox(value=False, label="Adaptive mode (tune ĂŽÂ±, ÄŽâ€°)")
                cycles       = gr.Slider(1, 6,   value=3,  step=1,     label="Cycles")
                alpha_min    = gr.Slider(0.02, 0.30, value=0.04, step=0.01,  label="ĂŽÂ± min")
                alpha_max    = gr.Slider(0.02, 0.30, value=0.16, step=0.01,  label="ĂŽÂ± max")
                alpha_step   = gr.Slider(0.005,0.05, value=0.02, step=0.005, label="ĂŽÂ± step")
            with gr.Row():
                omega_min    = gr.Slider(2.0, 12.0, value=4.0, step=0.1, label="ÄŽâ€° min")
                omega_max    = gr.Slider(2.0, 12.0, value=9.5, step=0.1, label="ÄŽâ€° max")
                omega_steps  = gr.Slider(3, 16, value=6, step=1, label="ÄŽâ€° steps")
                target       = gr.Dropdown(choices=["maximize_rsig","maximize_r2","maximize_rsi"], value="maximize_rsig", label="Target")

            #--- Meta-Agent UI (znotraj with gr.Tab("Meta-Agent"): ...) ---

            # ... (zgornji kontrolniki: provider/chat_m/emb_m/... target) ...

            task = gr.Textbox(lines=4, label="Task", placeholder="Short task for the researcher agent...")
            run_agent_btn = gr.Button("Run Agent")
            ask_meta_btn  = gr.Button("Ask (Meta)", variant="secondary")

            diag2 = gr.HTML()
            with gr.Row():
                steps2_out  = gr.Code(label="Agent Steps")
                critic_out  = gr.Code(label="Critic Notes")

            # >>> USTVARI FINAL_OUT PREDEN GA UPORABIĹ  V outputs
            with gr.Accordion("đź§  Final Answer", open=True):
                final_out = gr.Markdown(label="Final Answer", value="")

            with gr.Row():
                rsi2_out = gr.Code(label="RSI result")
                ism2_out = gr.Code(label="ISM phase (Ď‰, Ď†, RÂ˛)")
            with gr.Row():
                plot_phi2  = gr.LinePlot(x="x", y="Φ(t)",  label="Φ(t)")
                plot_dphi2 = gr.LinePlot(x="x", y="∂Φ/∂t", label="∂Φ/∂t")
            trace2_dl  = gr.File(label="Download agent_trace.json")
            report2_dl = gr.File(label="Download agent_report.json")

            # (opcijsko) memory panel
            with gr.Accordion("đź§  Memory (recent context)", open=False):
                memory_box   = gr.Textbox(label="Previous memory", interactive=False, lines=5)
                use_mem_cb   = gr.Checkbox(value=False, label="Use memory as context for next question")
                clear_mem_btn= gr.Button("Clear memory")

            # --- Run Agent: klik (zdaj `final_out` Ĺľe obstaja)
            run_agent_btn.click(
                fn=run_meta_agent,
                inputs=[
                    provider, chat_m, emb_m, ollama_url,
                    task, steps2, mode2, alpha2,
                    adaptive_cb, cycles, alpha_min, alpha_max, alpha_step,
                    omega_min, omega_max, omega_steps, target
                ],
                outputs=[
                    diag2, steps2_out, critic_out, rsi2_out, ism2_out,
                    plot_phi2, plot_dphi2, trace2_dl, report2_dl,  # prvih 9
                    final_out                                      # 10. = final answer
                ]
            )

            # --- Memory gumba + Ask (Meta)
            clear_mem_btn.click(fn=lambda: "", inputs=[], outputs=[memory_box])

            ask_meta_btn.click(
                fn=run_meta_ask,
                inputs=[
                    provider, chat_m, emb_m, ollama_url,
                    task, steps2, mode2, alpha2,
                    adaptive_cb, cycles, alpha_min, alpha_max, alpha_step,
                    omega_min, omega_max, omega_steps, target,
                    memory_box, use_mem_cb
                ],
                outputs=[
                    diag2, steps2_out, critic_out, rsi2_out, ism2_out,
                    plot_phi2, plot_dphi2, trace2_dl, report2_dl, final_out,
                    memory_box
                ]
            )
    demo.launch()

if __name__ == "__main__":
    main()


