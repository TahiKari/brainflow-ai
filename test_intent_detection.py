"""Tests unitaires : détection d'intention (`detect_intent`) + intégration workflow.

Vérifie :
    1. Les mots-clés "direct"   → intent_type == "direct"
    2. Les mots-clés "analysis" → intent_type == "analysis"
    3. Un message neutre         → intent_type == "direct" (défaut)
    4. Case-insensitive + variantes orthographiques
    5. Le workflow propage bien `intent_type` dans le dict retourné
       (succès ET échec planner)
    6. Aucun autre champ du workflow n'est impacté (non-régression)
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

# On précharge `main` pour résoudre le graphe d'imports (router.__init__ →
# chat_routes → services.workflow) dans le bon ordre et éviter l'import
# circulaire historique qui apparaît si l'on importe `services.workflow`
# directement en premier.
import main  # noqa: F401  (import pour effet de bord : initialise le graphe)

from services import workflow as workflow_mod  # noqa: E402
detect_intent = workflow_mod.detect_intent
run_workflow = workflow_mod.run_workflow


# ---------------------------------------------------------------------------
# 1. Tests unitaires purs sur `detect_intent`
# ---------------------------------------------------------------------------

class TestDetectIntent(unittest.TestCase):
    """`detect_intent` est déterministe et ne fait aucun I/O."""

    # --- Mots-clés "direct" ----------------------------------------------
    def test_direct_repond(self):
        self.assertEqual(detect_intent("Répond à cette question."), "direct")

    def test_direct_ecris(self):
        self.assertEqual(detect_intent("Écris un poème."), "direct")

    def test_direct_genere(self):
        self.assertEqual(detect_intent("Génère une liste de 10 idées."), "direct")

    def test_direct_fais(self):
        self.assertEqual(detect_intent("Fais un résumé."), "direct")

    # --- Mots-clés "analysis" --------------------------------------------
    def test_analysis_explique(self):
        self.assertEqual(detect_intent("Explique la récursion."), "analysis")

    def test_analysis_analyse(self):
        self.assertEqual(detect_intent("Analyse ce jeu de données."), "analysis")

    def test_analysis_pourquoi(self):
        self.assertEqual(detect_intent("Pourquoi le ciel est bleu ?"), "analysis")

    def test_analysis_comment(self):
        self.assertEqual(detect_intent("Comment fonctionne HTTP ?"), "analysis")

    # --- Défaut -----------------------------------------------------------
    def test_default_neutral_message(self):
        self.assertEqual(detect_intent("Bonjour."), "direct")

    def test_default_empty_string(self):
        self.assertEqual(detect_intent(""), "direct")

    def test_default_non_string(self):
        # Robustesse : None / int / dict → "direct"
        self.assertEqual(detect_intent(None), "direct")            # type: ignore[arg-type]
        self.assertEqual(detect_intent(42), "direct")              # type: ignore[arg-type]
        self.assertEqual(detect_intent({"x": 1}), "direct")        # type: ignore[arg-type]

    # --- Case-insensitivity ----------------------------------------------
    def test_case_insensitive_direct(self):
        self.assertEqual(detect_intent("ECRIS un résumé."), "direct")
        self.assertEqual(detect_intent("GÉNÈRE 3 titres"), "direct")

    def test_case_insensitive_analysis(self):
        self.assertEqual(detect_intent("EXPLIQUE la relativité."), "analysis")
        self.assertEqual(detect_intent("POURQUOI ça marche ?"), "analysis")

    # --- Priorité : direct > analysis si les deux sont présents ----------
    def test_direct_wins_over_analysis_when_both_present(self):
        msg = "Explique pourquoi, puis écris un résumé."
        # "écris" → direct doit gagner
        self.assertEqual(detect_intent(msg), "direct")

    # --- Sous-chaînes (les mots-clés sont des substrings) ----------------
    def test_substring_match(self):
        # "répondre" contient "répond" → direct
        self.assertEqual(detect_intent("Peux-tu répondre ?"), "direct")
        # "analysez" contient "analyse" → analysis
        self.assertEqual(detect_intent("Analysez ces résultats."), "analysis")

    # --- Ponctuation / markdown ------------------------------------------
    def test_with_markdown_and_punctuation(self):
        self.assertEqual(detect_intent("**Analyse** le code."), "analysis")
        self.assertEqual(detect_intent("-> Fais ceci !"), "direct")


# ---------------------------------------------------------------------------
# 2. Intégration workflow : `intent_type` présent dans la réponse
# ---------------------------------------------------------------------------

class TestWorkflowIntegration(unittest.TestCase):
    """Vérifie que `run_workflow` renvoie bien `intent_type` dans tous les cas."""

    def _run(self, coro):
        return asyncio.run(coro)

    @patch.object(workflow_mod, "plan_task", new_callable=AsyncMock)
    def test_intent_type_present_on_planner_failure(self, mock_plan):
        mock_plan.return_value = {"ok": False, "error": "boom"}

        result = self._run(run_workflow("Explique la récursion."))

        self.assertFalse(result["ok"])
        self.assertEqual(result["intent_type"], "analysis")
        self.assertEqual(result["plan"], [])
        self.assertEqual(result["summary"]["total"], 0)
        print("[OK] intent_type propagé sur échec planner (=analysis)")

    @patch.object(workflow_mod, "plan_task", new_callable=AsyncMock)
    def test_intent_type_present_on_empty_subtasks(self, mock_plan):
        """Mode analysis avec sous-tâches vides → retour ok=False, intent="analysis".

        Note : en mode "direct" le planner est court-circuité, donc ce scénario
        n'existe plus que pour `intent_type == "analysis"`.
        """
        mock_plan.return_value = {"ok": True, "subtasks": []}

        result = self._run(run_workflow("Explique-moi ce sujet."))

        self.assertFalse(result["ok"])
        self.assertEqual(result["intent_type"], "analysis")
        self.assertEqual(result["plan"], [])
        mock_plan.assert_awaited_once()
        print("[OK] intent_type propagé sur sous-tâches vides (=analysis)")

    @patch.object(workflow_mod, "validate_response", new_callable=AsyncMock)
    @patch.object(workflow_mod, "_dispatch_agent")
    @patch.object(workflow_mod, "plan_task", new_callable=AsyncMock)
    def test_intent_type_present_on_full_success(
        self, mock_plan, mock_dispatch, mock_validate
    ):
        """Succès complet en mode analysis : planner + agent + validator."""
        mock_plan.return_value = {
            "ok": True,
            "subtasks": [
                {"id": 1, "description": "Expliquer le sujet.",
                 "task_type": "moyen"}
            ],
        }

        async def _agent_coro():
            return {"ok": True, "text": "Voici l'explication.", "used": "mock"}

        mock_dispatch.return_value = _agent_coro()
        mock_validate.return_value = {
            "ok": True, "valid": True, "score": 90, "used": "mock"
        }

        # Message "analysis" (contient "explique") → pipeline complet activé
        result = self._run(run_workflow("Explique-moi la récursion."))

        # Cohérence sur tous les champs clés
        self.assertTrue(result["ok"])
        self.assertEqual(result["intent_type"], "analysis")
        self.assertEqual(len(result["plan"]), 1)
        self.assertEqual(result["summary"]["total"], 1)
        self.assertEqual(result["summary"]["succeeded"], 1)
        self.assertEqual(result["summary"]["validated"], 1)
        print("[OK] intent_type propagé sur succès complet (=analysis)")

    @patch.object(workflow_mod, "validate_response", new_callable=AsyncMock)
    @patch.object(workflow_mod, "_dispatch_agent")
    @patch.object(workflow_mod, "plan_task", new_callable=AsyncMock)
    def test_non_regression_original_fields_preserved(
        self, mock_plan, mock_dispatch, mock_validate
    ):
        """L'ajout de `intent_type` ne casse aucun champ existant (mode analysis)."""
        mock_plan.return_value = {
            "ok": True,
            "subtasks": [
                {"id": 1, "description": "Analyser le sujet.",
                 "task_type": "moyen"}
            ],
        }

        async def _agent_coro():
            return {"ok": True, "text": "Analyse fictive.", "used": "mock"}

        mock_dispatch.return_value = _agent_coro()
        mock_validate.return_value = {
            "ok": True, "valid": True, "score": 80, "used": "mock"
        }

        # Mot-clé "analyse" → pipeline analysis (champs historiques pertinents)
        result = self._run(run_workflow("Analyse ces chiffres."))

        # Tous les champs historiques sont toujours présents
        for key in ("ok", "user_input", "plan", "steps", "summary"):
            self.assertIn(key, result, f"champ manquant : {key}")

        # Le champ step conserve sa structure
        self.assertEqual(len(result["steps"]), 1)
        step = result["steps"][0]
        for key in ("id", "description", "task_type", "agent", "validation"):
            self.assertIn(key, step, f"champ step manquant : {key}")

        print("[OK] non-régression : tous les champs historiques préservés")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- detect_intent (unitaires) ---")
    suite1 = unittest.defaultTestLoader.loadTestsFromTestCase(TestDetectIntent)
    r1 = unittest.TextTestRunner(verbosity=0).run(suite1)

    print("\n--- Intégration workflow ---")
    suite2 = unittest.defaultTestLoader.loadTestsFromTestCase(TestWorkflowIntegration)
    r2 = unittest.TextTestRunner(verbosity=0).run(suite2)

    total_failures = len(r1.failures) + len(r1.errors) + len(r2.failures) + len(r2.errors)
    if total_failures == 0:
        print("\n[OK] Tous les tests intent_detection sont passes")
    else:
        print(f"\n[KO] {total_failures} test(s) en echec")
        raise SystemExit(1)
