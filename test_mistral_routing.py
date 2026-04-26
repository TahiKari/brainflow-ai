"""Tests de non-régression pour `select_mistral_model` + intégration routing.

Utilise des mocks pour éviter tout appel réseau. Couvre :
- Le mapping `select_mistral_model`.
- La sélection du bon modèle Mistral via `route_request` selon le `task_type`.
- Le fallback Gemini → Mistral (Mistral répond OK).
- Non-régression sur les autres providers (`call_gemini` inchangé).

Exécution :
    python test_mistral_routing.py
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import patch

from services.llm_service import select_mistral_model
from router import router as router_mod


# ---------------------------------------------------------------------------
# Utilitaires mocks
# ---------------------------------------------------------------------------

def _ok_response(provider: str, model: str, text: str = "ok") -> Dict[str, Any]:
    return {
        "ok": True,
        "provider": provider,
        "model": model,
        "text": text,
        "raw": {},
    }


def _ko_response(provider: str, error: str = "mocked error") -> Dict[str, Any]:
    return {
        "ok": False,
        "provider": provider,
        "error": error,
        "quota_exceeded": False,
    }


# ---------------------------------------------------------------------------
# 1. Tests unitaires du mapping
# ---------------------------------------------------------------------------

def test_select_mistral_model_mapping() -> None:
    assert select_mistral_model("simple") == "ministral-8b-latest"
    assert select_mistral_model("moyen") == "open-mixtral-8x7b"
    assert select_mistral_model("medium") == "open-mixtral-8x7b"
    assert select_mistral_model("complexe") == "mistral-large-latest"
    assert select_mistral_model("complex") == "mistral-large-latest"
    assert select_mistral_model("large") == "mistral-large-latest"
    assert select_mistral_model(None) == "open-mixtral-8x7b"
    assert select_mistral_model("") == "open-mixtral-8x7b"
    assert select_mistral_model("inconnu") == "open-mixtral-8x7b"
    print("[OK] test_select_mistral_model_mapping")


# ---------------------------------------------------------------------------
# 2. Intégration : Mistral sélectionné selon task_type via route_request
# ---------------------------------------------------------------------------

async def _run_routing_with_mistral_success(task_type: str, expected_model: str) -> None:
    """Gemini KO → Mistral OK avec le modèle attendu selon task_type."""
    captured: Dict[str, Any] = {}

    async def fake_gemini(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return _ko_response("gemini", "forced fallback")

    async def fake_mistral_by_task(
        tt: str, prompt: str, **kwargs: Any
    ) -> Dict[str, Any]:
        captured["task_type"] = tt
        model = select_mistral_model(tt)
        captured["model"] = model
        return _ok_response("mistral", model, text=f"réponse {model}")

    with patch.object(router_mod, "call_gemini", side_effect=fake_gemini), \
         patch.object(router_mod, "call_mistral_by_task", side_effect=fake_mistral_by_task):
        result = await router_mod.route_request(task_type, "ping")

    assert result["ok"] is True, f"routing KO : {result}"
    assert result["used"] == "mistral"
    assert result["response"]["model"] == expected_model
    assert captured["task_type"] == task_type
    assert captured["model"] == expected_model
    # Vérifie l'historique : 1 échec gemini puis succès mistral.
    providers = [a["provider"] for a in result["attempts"]]
    assert providers[0] == "gemini" and providers[1] == "mistral", providers
    print(f"[OK] routing task_type={task_type!r} -> {expected_model}")


async def test_routing_integration() -> None:
    await _run_routing_with_mistral_success("simple", "ministral-8b-latest")
    await _run_routing_with_mistral_success("moyen", "open-mixtral-8x7b")
    await _run_routing_with_mistral_success("complexe", "mistral-large-latest")
    # Tâche inconnue → défaut medium.
    await _run_routing_with_mistral_success("analyse", "open-mixtral-8x7b")


# ---------------------------------------------------------------------------
# 3. Non-régression : Gemini OK => on n'appelle jamais Mistral
# ---------------------------------------------------------------------------

async def test_gemini_success_shortcircuits_mistral() -> None:
    mistral_called = {"count": 0}

    async def fake_gemini(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return _ok_response("gemini", "gemini-flash-latest", text="réponse gemini")

    async def fake_mistral_by_task(tt: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        mistral_called["count"] += 1
        return _ok_response("mistral", "should-not-be-called")

    with patch.object(router_mod, "call_gemini", side_effect=fake_gemini), \
         patch.object(router_mod, "call_mistral_by_task", side_effect=fake_mistral_by_task):
        result = await router_mod.route_request("complexe", "ping")

    assert result["ok"] is True
    assert result["used"] == "gemini"
    assert mistral_called["count"] == 0, "Mistral ne doit pas être appelé si Gemini OK"
    print("[OK] test_gemini_success_shortcircuits_mistral")


# ---------------------------------------------------------------------------
# 4. Non-régression : tous providers KO => échec propre
# ---------------------------------------------------------------------------

async def test_all_providers_fail() -> None:
    async def ko(provider: str):
        async def _fn(prompt: str, **kwargs: Any) -> Dict[str, Any]:
            return _ko_response(provider)
        return _fn

    gemini_ko = await ko("gemini")
    mistral_ko_outer = await ko("mistral")
    groq_ko = await ko("groq")
    deepseek_ko = await ko("deepseek")
    openrouter_ko = await ko("openrouter")

    async def fake_mistral_by_task(tt: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return _ko_response("mistral")

    with patch.object(router_mod, "call_gemini", side_effect=gemini_ko), \
         patch.object(router_mod, "call_mistral_by_task", side_effect=fake_mistral_by_task), \
         patch.object(router_mod, "call_groq", side_effect=groq_ko), \
         patch.object(router_mod, "call_deepseek", side_effect=deepseek_ko), \
         patch.object(router_mod, "call_openrouter", side_effect=openrouter_ko):
        result = await router_mod.route_request("simple", "ping")

    assert result["ok"] is False
    assert result["used"] is None
    assert len(result["attempts"]) == 5
    print("[OK] test_all_providers_fail")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def _run_all() -> None:
    test_select_mistral_model_mapping()
    await test_routing_integration()
    await test_gemini_success_shortcircuits_mistral()
    await test_all_providers_fail()
    print("\nTous les tests sont passés ✅")


if __name__ == "__main__":
    asyncio.run(_run_all())
