# ------------------------------------------------------------------------------
# Harmonic RSI â€” Research Edition (v0.2.0)
# Licensed under CC BY-NC 4.0 (non-commercial use only)
# Â© Damjan Ĺ˝akelj 2025 â€” Harmonic Logos / ISM-X Framework
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Harmonic RSI â€” Research Edition (v0.2.0)
# Licensed under CC BY-NC 4.0 (non-commercial use only)
# Â© Freedom (Damjan Ĺ˝akelj) 2025 â€” Harmonic Logos / ISM-X Framework
# ------------------------------------------------------------------------------


import numpy as np
from harmonic_rsi import ISMField, ISMConfig, ResonanceEvaluator, ResonanceParams

def test_ism_rsi_pipeline():
    steps = ["Collect requirements", "Plan", "Implement", "Test", "Analyze", "Summarize"]
    emb = np.random.RandomState(0).randn(len(steps), 16).astype("float32")
    trace = {"steps": steps, "embeddings": emb.tolist()}
    ism = ISMField(ISMConfig(alpha=0.08, use_log_time=True)).fit(trace)
    est = ism.estimate_phase()
    ev  = ResonanceEvaluator()
    res = ev.evaluate(steps, mode="resonant",
                      params=ResonanceParams(alpha=0.08, omega=est["omega"], phi=est["phi"]))
    assert "resonance_signature" in res
    assert "omega" in est and "phi" in est and "r2" in est
