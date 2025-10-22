# ------------------------------------------------------------------------------
# Harmonic RSI — Research Edition (v0.2.0)
# Licensed under CC BY-NC 4.0 (non-commercial use only)
# © Damjan Žakelj 2025 — Harmonic Logos / ISM-X Framework
# ------------------------------------------------------------------------------

# harmonic_rsi/agents/__init__.py

from .providers import OpenAIProvider, OllamaProvider
from .researcher import ResearcherAgent
from .adaptive import AdaptiveResearcherAgent, AdaptConfig
from .harmonic_meta import HarmonicMeta, MetaConfig, FilterConfig

# Critic je opcijski (ÄŤe ga imaĹˇ v base.py â€“ super; ÄŤe ga ni, UI ne pade)
try:
    from .base import Critic  # noqa: F401
except Exception:
    Critic = None  # type: ignore

__all__ = [
    "OpenAIProvider", "OllamaProvider",
    "ResearcherAgent",
    "AdaptiveResearcherAgent", "AdaptConfig",
    "HarmonicMeta", "MetaConfig", "FilterConfig",
    # "Critic" je opcijski; ga ne navajamo v __all__ namerno
]
