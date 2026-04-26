"""Tests de routage intelligent vers Codestral.

Couvre :
- Le helper `_should_use_codestral` (matrice task_type × prompt).
- L'intégration : tâche de code → Codestral appelé en 1er, succès.
- Non-régression : chat normal → Codestral JAMAIS appelé, Gemini en 1er.
- Fallback : Codestral KO → Gemini prend la relève.

Exécution :
    python test_codestral_routing.py
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import patch

from router import router as router_mod
from router.router import _should_use_codestral


# ---------------------------------------------------------------------------
# Helpers mocks
# ---------------------------------------------------------------------------

def _ok(provider: str, model: str, text: str = "ok") -> Dict[str, Any]:
    return {"ok": True, "provider": provider, "model": model, "text": text, "raw": {}}


def _ko(provider: str, error: str = "mocked error") -> Dict[str, Any]:
    return {"ok": False, "provider": provider, "error": error, "quota_exceeded": False}


# ---------------------------------------------------------------------------
# 1. Helper _should_use_codestral
# ---------------------------------------------------------------------------

def test_should_use_codestral_rules() -> None:
    # task_type == "code" (case-insensitive)
    assert _should_use_codestral("code", "") is True
    assert _should_use_codestral("CODE", "bonjour") is True
    assert _should_use_codestral(" Code ", "salut") is True

    # Mots-clés dans le prompt (case-insensitive)
    assert _should_use_codestral(None, "Écris du code Python") is True
    assert _should_use_codestral("simple", "J'ai un BUG dans mon programme") is True
    assert _should_use_codestral("moyen", "help me with a python snippet") is True
    assert _should_use_codestral(None, "Write a bash SCRIPT") is True

    # Cas où Codestral NE doit PAS être utilisé
    assert _should_use_codestral(None, "Bonjour, comment vas-tu ?") is False
    assert _should_use_codestral("simple", "Quelle est la capitale de la France ?") is False
    assert _should_use_codestral("analyse", "Analyse ce texte marketing") is False
    assert _should_use_codestral("complexe", "Rédige un poème sur la mer") is False
    assert _should_use_codestral(None, "") is False
    assert _should_use_codestral(None, None) is False  # type: ignore[arg-type]

    print("[OK] test_should_use_codestral_rules")


# ---------------------------------------------------------------------------
# 2. Intégration : tâche de code → Codestral en 1er, succès
# ---------------------------------------------------------------------------

async def test_codestral_used_first_on_code_task() -> None:
    called = {"codestral": 0, "gemini": 0, "mistral": 0}

    async def fake_codestral(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        called["codestral"] += 1
        return _ok("codestral", "codestral-latest", text="def hello(): print('hi')")

    async def fake_gemini(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        called["gemini"] += 1
        return _ok("gemini", "gemini-flash-latest")

    async def fake_mistral_by_task(tt: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        called["mistral"] += 1
        return _ok("mistral", "x")

    with patch.object(router_mod, "call_codestral", side_effect=fake_codestral), \
         patch.object(router_mod, "call_gemini", side_effect=fake_gemini), \
         patch.object(router_mod, "call_mistral_by_task", side_effect=fake_mistral_by_task):
        # Déclenchement par task_type="code"
        r1 = await router_mod.route_request("code", "ping")
        # Déclenchement par mot-clé "python" dans le prompt
        r2 = await router_mod.route_request("simple", "Écris une fonction python")

    assert r1["ok"] is True and r1["used"] == "codestral"
    assert r2["ok"] is True and r2["used"] == "codestral"
    assert called["codestral"] == 2
    assert called["gemini"] == 0, "Gemini ne doit pas être appelé si Codestral OK"
    assert called["mistral"] == 0
    # Codestral est bien en 1re position dans l'historique
    assert r1["attempts"][0]["provider"] == "codestral"
    assert r2["attempts"][0]["provider"] == "codestral"
    print("[OK] test_codestral_used_first_on_code_task")


# ---------------------------------------------------------------------------
# 3. Non-régression : chat normal → Codestral JAMAIS appelé
# ---------------------------------------------------------------------------

async def test_codestral_never_called_on_chat() -> None:
    called = {"codestral": 0, "gemini": 0}

    async def fake_codestral(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        called["codestral"] += 1
        return _ok("codestral", "codestral-latest")

    async def fake_gemini(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        called["gemini"] += 1
        return _ok("gemini", "gemini-flash-latest", text="Bonjour !")

    with patch.object(router_mod, "call_codestral", side_effect=fake_codestral), \
         patch.object(router_mod, "call_gemini", side_effect=fake_gemini):
        r = await router_mod.route_request("simple", "Bonjour, comment vas-tu ?")

    assert r["ok"] is True
    assert r["used"] == "gemini", f"Gemini devait être utilisé, got {r['used']}"
    assert called["codestral"] == 0, "Codestral ne doit JAMAIS être appelé pour un chat normal"
    assert called["gemini"] == 1
    # Premier attempt = gemini (pas codestral)
    assert r["attempts"][0]["provider"] == "gemini"
    print("[OK] test_codestral_never_called_on_chat")


# ---------------------------------------------------------------------------
# 4. Fallback : Codestral KO → Gemini prend la relève
# ---------------------------------------------------------------------------

async def test_codestral_fallback_to_gemini_on_failure() -> None:
    called = {"codestral": 0, "gemini": 0}

    async def fake_codestral(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        called["codestral"] += 1
        return _ko("codestral", "boom")

    async def fake_gemini(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        called["gemini"] += 1
        return _ok("gemini", "gemini-flash-latest", text="ok fallback")

    with patch.object(router_mod, "call_codestral", side_effect=fake_codestral), \
         patch.object(router_mod, "call_gemini", side_effect=fake_gemini):
        r = await router_mod.route_request("code", "Fix this bug in my script")

    assert r["ok"] is True
    assert r["used"] == "gemini"
    assert called["codestral"] == 1
    assert called["gemini"] == 1
    providers = [a["provider"] for a in r["attempts"]]
    assert providers[:2] == ["codestral", "gemini"], providers
    print("[OK] test_codestral_fallback_to_gemini_on_failure")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def _run_all() -> None:
    test_should_use_codestral_rules()
    await test_codestral_used_first_on_code_task()
    await test_codestral_never_called_on_chat()
    await test_codestral_fallback_to_gemini_on_failure()
    print("\nTous les tests Codestral sont passés ✅")


if __name__ == "__main__":
    asyncio.run(_run_all())
