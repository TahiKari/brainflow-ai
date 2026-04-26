"""Test live : vérifie que `intent_type` est bien calculé et logué
par le serveur FastAPI en cours d'exécution.

Stratégie :
1. Envoie 3 requêtes POST /api/chat avec des prompts couvrant les 3
   branches de `detect_intent` (direct / analysis / défaut).
2. Vérifie que le serveur répond HTTP 200 pour chacune.
3. Affiche la réponse condensée pour inspection visuelle
   (intent_type lui-même est visible dans les logs serveur :
    `[brainflow.workflow] INFO | Intent détecté : ...`).

Utilisation :
    python test_intent_live.py
    (le serveur doit être démarré : python main.py)
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"

CASES = [
    ("direct  (écris)", "Écris une phrase courte de bienvenue."),
    ("analysis (explique)", "Explique brièvement le principe de la récursion."),
    ("défaut  (neutre)", "Bonjour, comment vas-tu ?"),
]


def _wait_for_health(retries: int = 20, delay: float = 0.5) -> bool:
    """Attend que /health réponde 200."""
    for i in range(retries):
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def _post_chat(message: str) -> dict:
    data = json.dumps({"message": message}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode("utf-8")
        return json.loads(body)


def main() -> int:
    print(f"=== Test live : intent_type via {BASE_URL}/api/chat ===")

    if not _wait_for_health():
        print("[KO] Serveur indisponible sur /health — démarre 'python main.py' d'abord.")
        return 1
    print("[OK] Serveur up.")

    ok_count = 0
    for label, message in CASES:
        print(f"\n--- {label} ---")
        print(f"  prompt : {message!r}")
        try:
            resp = _post_chat(message)
        except urllib.error.HTTPError as e:
            print(f"  [KO] HTTP {e.code} : {e.read().decode('utf-8', 'ignore')[:200]}")
            continue
        except Exception as e:
            print(f"  [KO] {type(e).__name__}: {e}")
            continue

        ok = resp.get("ok")
        summary = resp.get("summary", {})
        plan_len = len(resp.get("plan") or [])
        preview = (resp.get("response") or "")[:120].replace("\n", " ")

        print(f"  ok={ok}  plan={plan_len} steps  summary={summary}")
        print(f"  response preview: {preview!r}")
        if ok is not None:
            ok_count += 1

    print(f"\n=== {ok_count}/{len(CASES)} requêtes ont reçu une réponse HTTP valide ===")
    print("→ Vérifie les logs serveur : tu dois voir 3 lignes")
    print("  '[brainflow.workflow] INFO | Intent détecté : direct|analysis|direct'")
    return 0 if ok_count == len(CASES) else 2


if __name__ == "__main__":
    sys.exit(main())
