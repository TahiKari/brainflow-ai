"""Tests end-to-end : quota Gemini (429) → fallback Mistral via /api/chat.

Simule un quota dépassé sur Gemini et vérifie que le pipeline complet
(Planner → Router → Agent → Validator) continue de fonctionner via
Mistral (moyen), en passant par l'endpoint HTTP public `/api/chat`.

Approche : ASGITransport pour exécuter l'app FastAPI in-process.
Les appels LLM sont patchés (`patch.object` sur les attributs importés
dans les modules qui les consomment : `agents.planner`, `router.router`,
`agents.code_agent`, `agents.validator`).

Aucune clé API n'est nécessaire — tout est mocké.
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from unittest.mock import AsyncMock, patch

import httpx

# S'assurer que la racine du projet est importable.
sys.path.insert(0, ".")

from main import app  # noqa: E402

import agents.planner as planner_mod  # noqa: E402
import router.router as router_mod  # noqa: E402
import agents.code_agent as code_agent_mod  # noqa: E402
import agents.validator as validator_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLAN_JSON = json.dumps(
    [{"id": 1, "description": "Additionne 2 et 3", "task_type": "simple"}]
)


def _gemini_quota_response():
    return {
        "ok": False,
        "provider": "gemini",
        "error": "HTTP 429: Quota exceeded",
    }


def _mistral_plan_response():
    return {
        "ok": True,
        "provider": "mistral",
        "model": "open-mixtral-8x7b",
        "text": PLAN_JSON,
        "raw": {},
    }


def _mistral_answer_response():
    return {
        "ok": True,
        "provider": "mistral",
        "model": "open-mixtral-8x7b",
        "text": "La somme est 5.",
        "raw": {},
    }


def _validator_ok():
    return {"ok": True, "valid": True, "score": 95, "used": "mock"}


# ---------------------------------------------------------------------------
# Tests end-to-end
# ---------------------------------------------------------------------------

class TestPlannerE2EFallback(unittest.TestCase):
    """Test HTTP complet : POST /api/chat avec Gemini en 429."""

    def test_chat_endpoint_survives_gemini_quota(self) -> None:
        """Gemini 429 côté planner ET côté router → tout passe par Mistral."""

        # --- Mocks planner ------------------------------------------------
        # Gemini planner : 429
        planner_gemini_mock = AsyncMock(return_value=_gemini_quota_response())
        # Mistral planner : renvoie un plan JSON valide
        planner_mistral_mock = AsyncMock(return_value=_mistral_plan_response())

        # --- Mocks router (agent exécutant la sous-tâche 'simple') --------
        # Gemini router : 429 aussi → fallback sur Mistral dans router
        router_gemini_mock = AsyncMock(return_value=_gemini_quota_response())
        router_mistral_mock = AsyncMock(return_value=_mistral_answer_response())

        # --- Mock validator (évite appel LLM réel) ------------------------
        validator_mock = AsyncMock(return_value=_validator_ok())

        async def run() -> None:
            # On patche les références importées dans chaque module.
            with patch.object(planner_mod, "call_gemini", planner_gemini_mock), \
                 patch.object(planner_mod, "call_mistral_medium",
                              planner_mistral_mock), \
                 patch.object(router_mod, "call_gemini", router_gemini_mock), \
                 patch.object(router_mod, "call_mistral_by_task",
                              router_mistral_mock), \
                 patch.object(validator_mod, "validate_response",
                              validator_mock):

                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/chat",
                        json={"message": "Additionne 2 et 3"},
                        timeout=10.0,
                    )

            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            self.assertTrue(data.get("ok"), f"Réponse KO : {data}")
            self.assertIn("response", data)
            self.assertTrue(
                data["response"].strip(),
                "La réponse doit contenir du texte.",
            )

            # Le planner doit avoir tenté Gemini (1x) puis Mistral (1x).
            planner_gemini_mock.assert_awaited_once()
            planner_mistral_mock.assert_awaited_once()

            # Le router doit également avoir été sollicité (Gemini échoue
            # puis Mistral réussit).
            self.assertGreaterEqual(router_gemini_mock.await_count, 1)
            self.assertGreaterEqual(router_mistral_mock.await_count, 1)

            # Le plan doit contenir exactement 1 sous-tâche.
            self.assertEqual(len(data.get("plan", [])), 1)
            self.assertEqual(data["plan"][0]["task_type"], "simple")

            print(
                "[OK] /api/chat E2E | "
                f"planner_gemini={planner_gemini_mock.await_count} "
                f"planner_mistral={planner_mistral_mock.await_count} "
                f"router_gemini={router_gemini_mock.await_count} "
                f"router_mistral={router_mistral_mock.await_count}"
            )

        asyncio.run(run())

    def test_chat_endpoint_all_planner_providers_fail(self) -> None:
        """Si Gemini + Mistral + Groq échouent tous au planner, l'endpoint
        renvoie une erreur propre (HTTP 200, ok=False) sans exception."""

        err = {"ok": False, "provider": "x", "error": "HTTP 500: down"}

        async def run() -> None:
            with patch.object(planner_mod, "call_gemini",
                              AsyncMock(return_value=err)), \
                 patch.object(planner_mod, "call_mistral_medium",
                              AsyncMock(return_value=err)), \
                 patch.object(planner_mod, "call_groq",
                              AsyncMock(return_value=err)):

                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/chat",
                        json={"message": "test"},
                        timeout=10.0,
                    )

            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            self.assertFalse(data.get("ok"))
            self.assertIn("error", data)
            self.assertIn("Planner", data["error"])
            print(f"[OK] /api/chat all_planner_failed | error={data['error'][:100]}")

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- E2E /api/chat : quota Gemini → fallback Mistral ---")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPlannerE2EFallback)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("\nTous les tests E2E planner sont passés ✅")
