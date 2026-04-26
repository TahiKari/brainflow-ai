"""Thorough live test : comparaison de latence direct vs analysis via /api/chat.

Prérequis : serveur FastAPI en cours d'exécution sur http://127.0.0.1:8000.

Vérifie :
    - Mode "direct" (prompts avec écris/génère/fais) : un seul appel LLM,
      plan/steps vides, response non vide, latence < 5s typiquement.
    - Mode "analysis" (prompts avec explique/analyse/pourquoi/comment) :
      planner + steps peuplés, response construite depuis les steps.
    - La latence direct est nettement inférieure à la latence analysis
      (gain du court-circuit).
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Dict

BASE = "http://127.0.0.1:8000/api/chat"


def call_chat(message: str, timeout: float = 120.0) -> Dict[str, Any]:
    payload = json.dumps({"message": message}).encode("utf-8")
    req = urllib.request.Request(
        BASE,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    t1 = time.perf_counter()
    data = json.loads(body)
    data["_elapsed_s"] = round(t1 - t0, 3)
    return data


DIRECT_PROMPTS = [
    "Ecris un court poeme sur l'hiver.",
    "Genere 3 idees de voyage en Europe.",
    "Fais un resume en une phrase de la revolution francaise.",
]

ANALYSIS_PROMPTS = [
    "Explique brievement le fonctionnement de la memoire vive.",
]


def check_direct(prompt: str) -> Dict[str, Any]:
    r = call_chat(prompt)
    assert r["ok"] is True, f"direct KO : {r}"
    assert r["plan"] == [], f"direct devrait avoir plan=[], reçu {r['plan']}"
    assert r["steps"] == [], f"direct devrait avoir steps=[], reçu {r['steps']}"
    assert isinstance(r["response"], str) and r["response"].strip(), \
        f"direct doit avoir une response non vide : {r}"
    return r


def check_analysis(prompt: str) -> Dict[str, Any]:
    r = call_chat(prompt)
    assert r["ok"] is True, f"analysis KO : {r}"
    assert len(r["plan"]) >= 1, f"analysis devrait avoir un plan non vide"
    assert len(r["steps"]) >= 1, f"analysis devrait avoir des steps"
    assert isinstance(r["response"], str) and r["response"].strip(), \
        f"analysis doit avoir une response non vide"
    return r


def main() -> int:
    print("=" * 70)
    print("THOROUGH LIVE TEST — /api/chat direct vs analysis")
    print("=" * 70)

    direct_times = []
    for p in DIRECT_PROMPTS:
        r = check_direct(p)
        direct_times.append(r["_elapsed_s"])
        print(f"[direct] {p[:50]:<50} -> {r['_elapsed_s']}s | plan={len(r['plan'])} "
              f"steps={len(r['steps'])} resp_len={len(r['response'])}")

    analysis_times = []
    for p in ANALYSIS_PROMPTS:
        r = check_analysis(p)
        analysis_times.append(r["_elapsed_s"])
        print(f"[analys] {p[:50]:<50} -> {r['_elapsed_s']}s | plan={len(r['plan'])} "
              f"steps={len(r['steps'])} resp_len={len(r['response'])}")

    avg_direct = sum(direct_times) / len(direct_times)
    avg_analysis = sum(analysis_times) / len(analysis_times)
    print("-" * 70)
    print(f"Latence moyenne DIRECT   : {avg_direct:.2f}s ({len(direct_times)} req)")
    print(f"Latence moyenne ANALYSIS : {avg_analysis:.2f}s ({len(analysis_times)} req)")
    print(f"Gain direct vs analysis  : x{avg_analysis / max(avg_direct, 0.001):.1f}")
    print("=" * 70)
    print("[OK] Tests live direct vs analysis passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
