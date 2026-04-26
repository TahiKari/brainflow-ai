"""Test live manuel : appelle /api/chat sur le serveur local et vérifie
que le planner fonctionne de bout en bout avec les vrais providers.

Pré-requis :
    - Serveur FastAPI lancé : `python main.py` (port 8000).
    - Au moins une clé API valide parmi : GEMINI / MISTRAL / GROQ dans
      `api_keys.json`.

Ce script ne mocke rien : il vérifie l'intégration réelle. Il est tolérant
aux timeouts / erreurs réseau (log en cas d'échec, pas de crash).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:8000"


def _post(path: str, body: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url=f"{BASE_URL}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(path: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    # 1) Santé du serveur
    print(f"→ GET {BASE_URL}/health")
    try:
        health = _get("/health")
    except Exception as exc:
        print(f"[KO] Serveur injoignable : {exc}")
        print("Lance `python main.py` dans un autre terminal.")
        return 1
    print(f"   {health}")
    if health.get("status") != "ok":
        print("[KO] Health non OK.")
        return 1

    # 2) Liste des clés API configurées (présence minimale)
    print(f"→ GET {BASE_URL}/api/config/keys")
    keys = _get("/api/config/keys").get("keys", {})
    configured = [k for k, v in keys.items() if v]
    print(f"   Clés configurées : {configured or 'AUCUNE'}")
    if not configured:
        print("[SKIP] Aucune clé — on s'arrête là (test live nécessite ≥1 clé).")
        return 0

    # 3) Appel réel /api/chat sur une requête simple
    prompt = "Dis bonjour en une phrase."
    print(f"→ POST {BASE_URL}/api/chat  message='{prompt}'")
    t0 = time.perf_counter()
    try:
        data = _post("/api/chat", {"message": prompt}, timeout=90.0)
    except urllib.error.URLError as exc:
        print(f"[KO] Erreur réseau : {exc}")
        return 1
    dur_ms = (time.perf_counter() - t0) * 1000
    print(f"   HTTP OK en {dur_ms:.0f} ms")

    # Affichage synthétique
    ok = bool(data.get("ok"))
    plan = data.get("plan") or []
    response = (data.get("response") or "").strip()
    summary = data.get("summary") or {}
    err = data.get("error")

    print(f"   ok={ok}  plan_len={len(plan)}  summary={summary}")
    if plan:
        print(f"   plan[0]={plan[0]}")
    if response:
        excerpt = response[:160].replace("\n", " ")
        print(f"   response≈ {excerpt!r}")
    if err:
        print(f"   error={err[:200]}")

    if not ok:
        print("[WARN] Workflow a répondu ok=False (possiblement quota dépassé "
              "sur le 1er provider) — le fallback a joué son rôle si on a "
              "au moins une réponse textuelle.")
        return 0 if response else 1

    print("[OK] Chat live fonctionne ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
