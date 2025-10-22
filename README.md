
# Harmonic RSI — open-core (Research Edition)

**Version:** 0.2.0  
**Author:** Damjan Žakelj (Harmonic Logos / ISM-X Framework)  
**License:** CC BY-NC 4.0 — free for research, education and personal non-commercial use  

---

### About
Harmonic RSI measures resonance and stability in reasoning agents.  
This *Research Edition* provides the open-core modules and meta-agent layer under a non-commercial license.  
Commercial or enterprise deployment requires written permission from the author.

---

### License
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)  
Use for research, education and personal projects is free.  
Commercial or enterprise use requires written permission from the author.

Full text: https://creativecommons.org/licenses/by-nc/4.0/

## Quick Start (Windows PowerShell)
```powershell
# 1) Create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install (editable)
pip install -e .

# 3) Run the Gradio app
harmonic-rsi-app
# or:
python -m harmonic_rsi.app_gradio
```

### OpenAI or Ollama
- **OpenAI**: set `OPENAI_API_KEY` env var or enter it in UI textbox.
- **Ollama**: run `ollama serve` locally, e.g. `ollama run llama3.1:8b`.
  - Embeddings: configure a local embedding model (e.g. `nomic-embed-text`).

## Agents CLI
```powershell
# OpenAI
python -m harmonic_rsi.agents.run_agent --task "Zakaj je nebo modro?" --provider openai

# Ollama
python -m harmonic_rsi.agents.run_agent --task "Why is the sky blue?" --provider ollama --ollama_url http://localhost:11434
```

Outputs:
- `agent_trace.json` (steps + embeddings), `agent_report.json` (rsi, ism, final_text), and `agent_final.txt`.

## Python API (stable surface)
```python
from harmonic_rsi import ISMField, ISMConfig, ResonanceEvaluator, ResonanceParams

trace = {"steps": ["Collect", "Plan", "Implement", "Test"], "embeddings": [[0.1]*8]*4}
ism = ISMField(ISMConfig(alpha=0.08, use_log_time=True)).fit(trace)
est = ism.estimate_phase()   # {'omega': ..., 'phi': ..., 'r2': ...}

ev = ResonanceEvaluator()
r = ev.evaluate(trace["steps"], mode="resonant",
                params=ResonanceParams(alpha=0.08, omega=est["omega"], phi=est["phi"]))
print(r)
```

## Project Norms (Harmonic Logos register)
- **Goals (Oct 2025):** open‑core RSI v1.0 → Φ‑layer (ISM) → σ‑clock (EIW) → integrated controller.
- **DoD v0.2.0:** app runs, agents CLI runs, API stable, minimal tests pass (no external keys needed).
- **OS/Tooling:** Windows‑first instructions; avoid hard `/mnt/data` paths; PowerShell‑friendly.
- **Truth Protocol:** be blunt, cite when needed, and mark uncertainty.

## Dev & Tests
```powershell
pip install -e ".[dev]"
pytest -q
```
CI uses GitHub Actions (python 3.9–3.12).

## Privacy/Keys
- Never commit `memory.jsonl` or keys.
- OpenAI usage: no chain‑of‑thought is stored; only final steps/answers.
