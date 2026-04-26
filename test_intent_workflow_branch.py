"""Tests : branchement de `run_workflow` selon `intent_type`.

Vérifie que :
1. intent_type == "direct"  → route_request("simple", ...) appelé UNE fois,
   planner/validator NON appelés, dict retourné avec `response`, `plan=[]`,
   `steps=[]`.
2. intent_type == "analysis" → comportement inchangé : planner + dispatch
   + validator appelés, `steps` peuplés, `plan` peuplé.
3. Erreur en mode direct → ok=False, error présent, pas de crash.
4. Route chat_routes : forward correct du champ `response` en mode direct.
5. Non-régression : route chat_routes fonctionne encore en mode analysis.

Exécution :
    python test_intent_workflow_branch.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from unittest.mock import AsyncMock, patch

# Contournement de l'import circulaire historique (cf. test_intent_detection.py).
import main  # noqa: F401  (précharge l'app FastAPI)

from services import workflow as workflow_mod
from services.workflow import run_workflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# 1) Mode DIRECT
# ---------------------------------------------------------------------------

class TestDirectMode(unittest.TestCase):
    """En mode direct : un seul appel LLM, pas de planner/validator."""

    def test_direct_calls_only_route_request_once(self):
        """Message 'écris ...' → planner non appelé, route_request appelé 1 fois."""
        with patch.object(workflow_mod, "route_request",
                          new=AsyncMock(return_value={
                              "ok": True,
                              "text": "Bienvenue chez nous !",
                              "used": "groq",
                          })) as m_route, \
             patch.object(workflow_mod, "plan_task",
                          new=AsyncMock()) as m_plan, \
             patch.object(workflow_mod, "validate_response",
                          new=AsyncMock()) as m_valid:

            result = _run(run_workflow("Écris une phrase de bienvenue."))

            # Planner et validator NON appelés en mode direct
            m_plan.assert_not_called()
            m_valid.assert_not_called()
            # route_request appelé UNE SEULE fois avec task_type="simple"
            self.assertEqual(m_route.call_count, 1)
            args, kwargs = m_route.call_args
            # Appel positionnel : ("simple", user_input)
            self.assertEqual(args[0], "simple")
            self.assertIn("bienvenue", args[1].lower())

        # Dict de retour : structure attendue
        self.assertTrue(result["ok"])
        self.assertEqual(result["intent_type"], "direct")
        self.assertEqual(result["response"], "Bienvenue chez nous !")
        self.assertEqual(result["plan"], [])
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["summary"]["succeeded"], 1)
        self.assertEqual(result["summary"]["failed"], 0)
        self.assertEqual(result["summary"]["total"], 0)
        self.assertNotIn("error", result)

    def test_direct_default_intent_also_bypasses_planner(self):
        """Message neutre → intent défaut 'direct' → toujours bypass du planner."""
        with patch.object(workflow_mod, "route_request",
                          new=AsyncMock(return_value={
                              "ok": True,
                              "text": "Salut !",
                              "used": "gemini",
                          })), \
             patch.object(workflow_mod, "plan_task",
                          new=AsyncMock()) as m_plan:

            result = _run(run_workflow("Bonjour"))

            m_plan.assert_not_called()

        self.assertEqual(result["intent_type"], "direct")
        self.assertEqual(result["response"], "Salut !")

    def test_direct_route_request_failure(self):
        """route_request échoue → ok=False, error présent, pas de crash."""
        with patch.object(workflow_mod, "route_request",
                          new=AsyncMock(return_value={
                              "ok": False,
                              "error": "Tous les providers ont échoué.",
                          })), \
             patch.object(workflow_mod, "plan_task",
                          new=AsyncMock()) as m_plan:

            result = _run(run_workflow("Écris quelque chose"))

            m_plan.assert_not_called()

        self.assertFalse(result["ok"])
        self.assertEqual(result["intent_type"], "direct")
        self.assertEqual(result["response"], "")
        self.assertIn("error", result)
        self.assertIn("providers", result["error"].lower())
        self.assertEqual(result["summary"]["failed"], 1)

    def test_direct_route_request_raises(self):
        """route_request lève une exception → capturée, ok=False."""
        with patch.object(workflow_mod, "route_request",
                          new=AsyncMock(side_effect=RuntimeError("boom"))), \
             patch.object(workflow_mod, "plan_task",
                          new=AsyncMock()) as m_plan:

            result = _run(run_workflow("Écris un poème"))

            m_plan.assert_not_called()

        self.assertFalse(result["ok"])
        self.assertEqual(result["intent_type"], "direct")
        self.assertIn("error", result)
        self.assertIn("boom", result["error"])


# ---------------------------------------------------------------------------
# 2) Mode ANALYSIS (non-régression)
# ---------------------------------------------------------------------------

class TestAnalysisMode(unittest.TestCase):
    """En mode analysis : pipeline complet, aucun changement par rapport à avant."""

    def test_analysis_calls_planner_and_dispatches(self):
        """Message 'explique ...' → planner + dispatch + validator appelés."""
        plan_mock = AsyncMock(return_value={
            "ok": True,
            "subtasks": [
                {"id": 1, "task_type": "moyen", "description": "Définir la récursion"},
                {"id": 2, "task_type": "moyen", "description": "Donner un exemple"},
            ],
        })
        route_mock = AsyncMock(return_value={
            "ok": True,
            "text": "Réponse fictive.",
            "used": "mistral",
        })
        valid_mock = AsyncMock(return_value={
            "ok": True,
            "valid": True,
            "score": 9,
            "used": "groq",
        })

        with patch.object(workflow_mod, "plan_task", new=plan_mock), \
             patch.object(workflow_mod, "route_request", new=route_mock), \
             patch.object(workflow_mod, "validate_response", new=valid_mock):

            result = _run(run_workflow("Explique la récursion."))

        # Le planner a bien été appelé
        plan_mock.assert_awaited_once()
        # route_request appelé pour chaque sous-tâche (2)
        self.assertEqual(route_mock.call_count, 2)
        # validator appelé pour chaque succès (2)
        self.assertEqual(valid_mock.call_count, 2)

        # Structure retournée = schéma analysis classique
        self.assertTrue(result["ok"])
        self.assertEqual(result["intent_type"], "analysis")
        self.assertEqual(len(result["plan"]), 2)
        self.assertEqual(len(result["steps"]), 2)
        self.assertEqual(result["summary"]["total"], 2)
        self.assertEqual(result["summary"]["succeeded"], 2)
        self.assertEqual(result["summary"]["validated"], 2)
        # Pas de champ `response` en mode analysis
        self.assertNotIn("response", result)

    def test_analysis_planner_failure_unchanged(self):
        """Planner échoue en mode analysis → comportement historique préservé."""
        with patch.object(workflow_mod, "plan_task",
                          new=AsyncMock(return_value={
                              "ok": False,
                              "error": "Planner down",
                          })):

            result = _run(run_workflow("Explique pourquoi le ciel est bleu."))

        self.assertFalse(result["ok"])
        self.assertEqual(result["intent_type"], "analysis")
        self.assertIn("Planner", result["error"])
        self.assertEqual(result["plan"], [])
        self.assertEqual(result["steps"], [])
        self.assertNotIn("response", result)


# ---------------------------------------------------------------------------
# 3) Intégration chat_routes (HTTP)
# ---------------------------------------------------------------------------

class TestChatRoutesIntegration(unittest.TestCase):
    """Vérifie que POST /api/chat renvoie bien le champ `response` adapté."""

    def _post_chat(self, message: str):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        return client.post("/api/chat", json={"message": message})

    def test_direct_mode_http(self):
        """En mode direct : la réponse HTTP contient le texte et pas de steps."""
        with patch.object(workflow_mod, "route_request",
                          new=AsyncMock(return_value={
                              "ok": True,
                              "text": "Voici votre réponse directe.",
                              "used": "groq",
                          })), \
             patch.object(workflow_mod, "plan_task",
                          new=AsyncMock()) as m_plan:

            resp = self._post_chat("Écris une salutation.")

            m_plan.assert_not_called()

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["response"], "Voici votre réponse directe.")
        self.assertEqual(body["plan"], [])
        self.assertEqual(body["steps"], [])

    def test_analysis_mode_http(self):
        """En mode analysis : la réponse HTTP contient steps et response enrichi."""
        plan_mock = AsyncMock(return_value={
            "ok": True,
            "subtasks": [
                {"id": 1, "task_type": "moyen", "description": "Étape A"},
            ],
        })
        route_mock = AsyncMock(return_value={
            "ok": True,
            "text": "Détail étape A.",
            "used": "mistral",
        })
        valid_mock = AsyncMock(return_value={
            "ok": True, "valid": True, "score": 8, "used": "groq",
        })

        with patch.object(workflow_mod, "plan_task", new=plan_mock), \
             patch.object(workflow_mod, "route_request", new=route_mock), \
             patch.object(workflow_mod, "validate_response", new=valid_mock):

            resp = self._post_chat("Explique brièvement.")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        # response non vide (construit depuis steps)
        self.assertTrue(body["response"])
        self.assertIn("Détail étape A.", body["response"])
        # steps présents et peuplés
        self.assertEqual(len(body["steps"]), 1)
        self.assertEqual(len(body["plan"]), 1)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Verbose par défaut pour visibilité CI / local
    unittest.main(verbosity=2)
