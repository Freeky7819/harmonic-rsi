
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
