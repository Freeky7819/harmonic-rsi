# ------------------------------------------------------------------------------
# Harmonic RSI — Research Edition (v0.2.0)
# Licensed under CC BY-NC 4.0 (non-commercial use only)
# © Damjan Žakelj 2025 — Harmonic Logos / ISM-X Framework
# ------------------------------------------------------------------------------


from __future__ import annotations
import os, requests
from typing import List, Dict
from openai import OpenAI

class OpenAIProvider:
    def __init__(self, api_key: str|None=None, chat_model="gpt-4o", emb_model="text-embedding-3-large"):
        key = api_key or os.getenv("OPENAI_API_KEY") or ""
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=key)
        self.chat_model = chat_model
        self.emb_model = emb_model

    # ---------- low-level wrappers (already used elsewhere) ----------
    def chat(self, messages: List[Dict[str,str]], **kw) -> str:
        try:
            r = self.client.responses.create(model=self.chat_model, input=messages, **kw)
            return r.output[0].content[0].text
        except Exception:
            cc = self.client.chat.completions.create(model=self.chat_model, messages=messages, **kw)
            return cc.choices[0].message.content

    def embed(self, texts: List[str]) -> List[List[float]]:
        em = self.client.embeddings.create(model=self.emb_model, input=texts)
        return [e.embedding for e in em.data]

    # ---------- high-level helpers expected by ResearcherAgent ----------
    def get_steps(self, task: str, n: int) -> List[str]:
        """Produce a numbered list of n concrete steps (no chain-of-thought)."""
        sys = {"role": "system",
               "content": f"Respond with a numbered list of {n} concrete actions. "
                          f"Be concise. No chain-of-thought; only final steps."}
        user = {"role": "user", "content": task}
        text = self.chat([sys, user]).strip()
        # Parse numbered/bulleted list into clean steps
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cleaned: List[str] = []
        for ln in lines:
            # remove bullet/number prefixes
            ln2 = ln.lstrip("-* ").lstrip("0123456789").lstrip(".)").strip()
            cleaned.append(ln2 or ln)
        cleaned = [c for c in cleaned if c]
        # Fallback to whole text if parsing failed
        return (cleaned[:n] if cleaned else ([text] if text else []))

    def get_embeddings(self, steps: List[str]):
        """Return numpy array of embeddings for provided steps."""
        inputs = [f"[step {i+1}] {s}" for i, s in enumerate(steps)]
        vectors = self.embed(inputs)
        # Convert to numpy array (ResearcherAgent expects ndarray)
        import numpy as np
        return np.array(vectors, dtype="float32")

class OllamaProvider:
    def __init__(self, base_url="http://localhost:11434", chat_model="llama3.1:8b", emb_model="nomic-embed-text"):
        self.base = base_url.rstrip("/")
        self.chat_model = chat_model
        self.emb_model = emb_model

    # ---------- low-level ----------
    def chat(self, messages: List[Dict[str,str]], **kw) -> str:
        prompt = ""
        for m in messages:
            role = m.get("role","user")
            prompt += f"{role.upper()}: {m['content']}\n"
        payload = {"model": self.chat_model, "prompt": prompt, "stream": False}
        r = requests.post(f"{self.base}/api/generate", json=payload, timeout=600)
        r.raise_for_status()
        out = r.json().get("response","")
        return out

    def embed(self, texts: List[str]) -> List[List[float]]:
        r = requests.post(f"{self.base}/api/embeddings", json={"model": self.emb_model, "input": texts}, timeout=600)
        r.raise_for_status()
        data = r.json()
        if "embeddings" in data:
            return data["embeddings"]
        if "embedding" in data:
            return [data["embedding"]]
        raise RuntimeError("Unexpected Ollama embeddings response")

    # ---------- high-level helpers expected by ResearcherAgent ----------
    def get_steps(self, task: str, n: int) -> List[str]:
        """Produce a numbered list of n concrete steps from local model."""
        sys = {"role": "system",
               "content": f"Respond with a numbered list of {n} concrete actions. "
                          f"Be concise. No chain-of-thought; only final steps."}
        user = {"role": "user", "content": task}
        text = self.chat([sys, user]).strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cleaned: List[str] = []
        for ln in lines:
            ln2 = ln.lstrip("-* ").lstrip("0123456789").lstrip(".)").strip()
            cleaned.append(ln2 or ln)
        cleaned = [c for c in cleaned if c]
        return (cleaned[:n] if cleaned else ([text] if text else []))

    def get_embeddings(self, steps: List[str]):
        inputs = [f"[step {i+1}] {s}" for i, s in enumerate(steps)]
        vectors = self.embed(inputs)
        import numpy as np
        return np.array(vectors, dtype="float32")
