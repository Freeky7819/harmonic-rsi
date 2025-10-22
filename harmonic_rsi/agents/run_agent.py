# ------------------------------------------------------------------------------
# Harmonic RSI — Research Edition (v0.2.0)
# Licensed under CC BY-NC 4.0 (non-commercial use only)
# © Damjan Žakelj 2025 — Harmonic Logos / ISM-X Framework
# ------------------------------------------------------------------------------


import json, os
from .providers import OpenAIProvider, OllamaProvider
from .researcher import ResearcherAgent
from .adaptive import AdaptiveResearcherAgent, AdaptConfig



def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    p.add_argument("--chat_model", default="gpt-4o")
    p.add_argument("--emb_model", default="text-embedding-3-large")
    p.add_argument("--ollama_url", default="http://localhost:11434")
    p.add_argument("--task", required=True)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--alpha", type=float, default=0.08)
    args = p.parse_args()

    if args.provider == "openai":
        prov = OpenAIProvider(chat_model=args.chat_model, emb_model=args.emb_model)
    else:
        prov = OllamaProvider(base_url=args.ollama_url, chat_model=args.chat_model, emb_model=args.emb_model)

    agent = ResearcherAgent(prov, steps_n=args.steps, mode="resonant")
    out = agent.run(args.task, alpha=args.alpha)

    trace, report = out["trace"], out["report"]
    open("agent_trace.json", "w", encoding="utf-8").write(json.dumps(trace, ensure_ascii=False, indent=2))
    open("agent_report.json", "w", encoding="utf-8").write(json.dumps(report, ensure_ascii=False, indent=2))

    final_text = report.get("final_text", "")
    print("\nđź§  Final Answer:\n", final_text or "(none)")
    if final_text:
        open("agent_final.txt", "w", encoding="utf-8").write(final_text)

    print("[ok] wrote agent_trace.json, agent_report.json", "(+ agent_final.txt)" if final_text else "")

if __name__ == "__main__":
    main()
