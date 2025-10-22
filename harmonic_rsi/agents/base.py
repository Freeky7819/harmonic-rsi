# --- append in base.py -------------------------------------------------------
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Critic:
    """
    Minimalna implementacija kritika, da adaptive meta-agent dobi povratno informacijo.
    Vračamo kratek 'review' korakov. Namenjeno kot varna privzeta varianta.
    """
    style: str = "bullet"

    def review(self, steps: List[str], rsi: Optional[Dict[str, float]] = None) -> str:
        bullets = []
        if not steps:
            return "No steps to critique."

        # zelo enostavna heuristika
        if len(steps) < 4:
            bullets.append("Premalo korakov – poskusi razbiti nalogo na več manjših, konkretnih dejanj.")
        if any(len(s) < 8 for s in steps):
            bullets.append("Nekateri koraki so preveč kratki ali nejasni – dodaj glagole in merila uspeha.")
        if any(s.lower().startswith(("reflect", "razmisli")) for s in steps):
            bullets.append("Dodaj korake za preverjanje rezultatov (verifikacijo), ne le razmislek.")
        if rsi:
            coh = float(rsi.get("semantic_coherence", 0.0))
            if coh < 0.45:
                bullets.append("Nizka semantična koherenca – dodaj povzetek med koraki, da držiš rdečo nit.")
        # vedno izpiši nekaj konstruktivnih predlogov
        bullets.extend([
            "Naredi en verifikacijski korak (kaj bo dokaz, da je korak uspel).",
            "Povej, kateri viri ali orodja bodo uporabljeni (konkretno).",
        ])
        return "\n".join(f"- {b}" for b in bullets)

    # združljivost z morebitnimi klici .critique(...) ali klicnimi objekti
    def critique(self, steps: List[str], rsi: Optional[Dict[str, float]] = None) -> str:
        return self.review(steps, rsi)

    def __call__(self, steps: List[str], rsi: Optional[Dict[str, float]] = None) -> str:
        return self.review(steps, rsi)
