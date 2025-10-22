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


def test_import_core():
    import harmonic_rsi
    from harmonic_rsi import ISMField, ISMConfig, ResonanceEvaluator, ResonanceParams
    assert ISMField and ISMConfig and ResonanceEvaluator and ResonanceParams

def test_import_agents():
    from harmonic_rsi.agents import ResearcherAgent, AdaptiveResearcherAgent, AdaptConfig
    assert ResearcherAgent and AdaptiveResearcherAgent and AdaptConfig
