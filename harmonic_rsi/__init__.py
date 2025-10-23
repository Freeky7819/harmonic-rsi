# ------------------------------------------------------------------------------
# Harmonic RSI — Research Edition (v0.2.0)
# Licensed under CC BY-NC 4.0 (non-commercial use only)
# © Damjan Žakelj 2025 — Harmonic Logos / ISM-X Framework
# ------------------------------------------------------------------------------

# harmonic_rsi/__init__.py

from .rsi import ResonanceEvaluator, ResonanceParams
from .ism_field import ISMField, ISMConfig

__all__ = ["ResonanceEvaluator", "ResonanceParams", "ISMField", "ISMConfig"]

# Optional providers – only if available; not required for RSI/ISM tests
try:
    from .agents.providers import OpenAIProvider, OllamaProvider
    __all__ += ["OpenAIProvider", "OllamaProvider"]
except Exception:
    # keep package importable even if providers (or deps/keys) are missing
    pass
