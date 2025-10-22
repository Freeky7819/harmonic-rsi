# ------------------------------------------------------------------------------
# Harmonic RSI — Research Edition (v0.2.0)
# Licensed under CC BY-NC 4.0 (non-commercial use only)
# © Damjan Žakelj 2025 — Harmonic Logos / ISM-X Framework
# ------------------------------------------------------------------------------

# --- append in base.py -------------------------------------------------------
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Critic:
    """
    Minimalna implementacija kritika, da adaptive meta-agent dobi povratno informacijo.
    VraÄŤamo kratek 'review' korakov. Namenjeno kot varna privzeta varianta.
    """
    style: str = "bullet"

    def review(self, steps: List[str], rsi: Optional[Dict[str, float]] = None) -> str:
        bullets = []
        if not steps:
            return "No steps to critique."

        # zelo enostavna heuristika
        if len(steps) < 4:
            bullets.append("Premalo korakov â€“ poskusi razbiti nalogo na veÄŤ manjĹˇih, konkretnih dejanj.")
        if any(len(s) < 8 for s in steps):
            bullets.append("Nekateri koraki so preveÄŤ kratki ali nejasni â€“ dodaj glagole in merila uspeha.")
        if any(s.lower().startswith(("reflect", "razmisli")) for s in steps):
            bullets.append("Dodaj korake za preverjanje rezultatov (verifikacijo), ne le razmislek.")
        if rsi:
            coh = float(rsi.get("semantic_coherence", 0.0))
            if coh < 0.45:
                bullets.append("Nizka semantiÄŤna koherenca â€“ dodaj povzetek med koraki, da drĹľiĹˇ rdeÄŤo nit.")
        # vedno izpiĹˇi nekaj konstruktivnih predlogov
        bullets.extend([
            "Naredi en verifikacijski korak (kaj bo dokaz, da je korak uspel).",
            "Povej, kateri viri ali orodja bodo uporabljeni (konkretno).",
        ])
        return "\n".join(f"- {b}" for b in bullets)

    # zdruĹľljivost z morebitnimi klici .critique(...) ali klicnimi objekti
    def critique(self, steps: List[str], rsi: Optional[Dict[str, float]] = None) -> str:
        return self.review(steps, rsi)

    def __call__(self, steps: List[str], rsi: Optional[Dict[str, float]] = None) -> str:
        return self.review(steps, rsi)
