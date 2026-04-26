"""Tests du fallback Gemini → Mistral → Groq dans `agents.planner`.

Vérifie :
    - Succès direct Gemini → utilise Gemini, pas de fallback.
    - Erreur 429 Gemini → bascule sur Mistral (moyen).
    - Erreur Gemini + erreur Mistral → bascule sur Groq.
    - Tous en échec → dict d'erreur propre, pas d'exception.
    - Exception brute levée par un provider → capturée par try/except.
    - Détecteur `_is_quota_error` reconnaît les variantes (429, quota,
      rate limit, too many requests).
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

from agents import planner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


# Réponse JSON valide que le planner sait parser.
_VALID_JSON = (
    '[{"id": 1, "description": "Étape A", "task_type": "simple"},'
    ' {"id": 2, "description": "Étape B", "task_type": "code"}]'
)


def _ok(provider: str, text: str = _VALID_JSON) -> Dict[str, Any]:
    return {
        "ok": True,
        "provider": provider,
        "model": f"{provider}-mock",
        "text": text,
        "raw": {},
    }


def _err(provider: str, error: str) -> Dict[str, Any]:
    return {"ok": False, "provider": provider, "error": error}


# ---------------------------------------------------------------------------
# Tests du détecteur
# ---------------------------------------------------------------------------

class TestQuotaDetector(unittest.TestCase):
    """`_is_quota_error` doit reconnaître les erreurs 429/quota/rate-limit."""

    def test_detects_429(self):
        self.assertTrue(planner._is_quota_error("HTTP 429: Too Many Requests"))

    def test_detects_quota_exceeded(self):
        self.assertTrue(planner._is_quota_error("quota exceeded for today"))

    def test_detects_rate_limit(self):
        self.assertTrue(planner._is_quota_error("Rate limit reached for user"))
        self.assertTrue(planner._is_quota_error("error: rate_limit_exceeded"))

    def test_detects_too_many_requests(self):
        self.assertTrue(planner._is_quota_error("Too Many Requests"))

    def test_case_insensitive(self):
        self.assertTrue(planner._is_quota_error("QUOTA EXCEEDED"))

    def test_rejects_other_errors(self):
        self.assertFalse(planner._is_quota_error("HTTP 500: Internal Server Error"))
        self.assertFalse(planner._is_quota_error("bad request"))
        self.assertFalse(planner._is_quota_error(""))
        self.assertFalse(planner._is_quota_error(None))  # type: ignore

    def test_runs(self):
        print("[OK] TestQuotaDetector (5 checks)")


# ---------------------------------------------------------------------------
# Tests de fallback
# ---------------------------------------------------------------------------

class TestPlannerFallback(unittest.TestCase):
    """Vérifie l'ordre Gemini → Mistral (moyen) → Groq."""

    def test_gemini_success_no_fallback(self):
        """Gemini OK → utilisé, Mistral/Groq jamais appelés."""
        gemini_mock = AsyncMock(return_value=_ok("gemini"))
        mistral_mock = AsyncMock(return_value=_ok("mistral"))
        groq_mock = AsyncMock(return_value=_ok("groq"))

        with patch.object(planner, "call_gemini", gemini_mock), \
             patch.object(planner, "call_mistral_medium", mistral_mock), \
             patch.object(planner, "call_groq", groq_mock):
            result = _run(planner.plan_task("Hello world"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "gemini")
        self.assertEqual(len(result["subtasks"]), 2)
        gemini_mock.assert_awaited_once()
        mistral_mock.assert_not_awaited()
        groq_mock.assert_not_awaited()
        print("[OK] test_gemini_success_no_fallback")

    def test_gemini_429_fallback_to_mistral(self):
        """Gemini retourne HTTP 429 → Mistral (moyen) utilisé."""
        gemini_mock = AsyncMock(
            return_value=_err("gemini", "HTTP 429: quota exceeded")
        )
        mistral_mock = AsyncMock(return_value=_ok("mistral_medium"))
        groq_mock = AsyncMock(return_value=_ok("groq"))

        with patch.object(planner, "call_gemini", gemini_mock), \
             patch.object(planner, "call_mistral_medium", mistral_mock), \
             patch.object(planner, "call_groq", groq_mock):
            result = _run(planner.plan_task("Decompose this task"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "mistral_medium")
        self.assertEqual(len(result["subtasks"]), 2)
        gemini_mock.assert_awaited_once()
        mistral_mock.assert_awaited_once()
        groq_mock.assert_not_awaited()
        print("[OK] test_gemini_429_fallback_to_mistral")

    def test_gemini_and_mistral_fail_fallback_to_groq(self):
        """Gemini 429 + Mistral 500 → Groq utilisé en dernier recours."""
        gemini_mock = AsyncMock(
            return_value=_err("gemini", "HTTP 429: Too Many Requests")
        )
        mistral_mock = AsyncMock(
            return_value=_err("mistral", "HTTP 500: Internal Server Error")
        )
        groq_mock = AsyncMock(return_value=_ok("groq"))

        with patch.object(planner, "call_gemini", gemini_mock), \
             patch.object(planner, "call_mistral_medium", mistral_mock), \
             patch.object(planner, "call_groq", groq_mock):
            result = _run(planner.plan_task("Task planning"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "groq")
        gemini_mock.assert_awaited_once()
        mistral_mock.assert_awaited_once()
        groq_mock.assert_awaited_once()
        print("[OK] test_gemini_and_mistral_fail_fallback_to_groq")

    def test_all_providers_fail_returns_error_dict(self):
        """Tous en échec → `ok=False` avec détails, aucune exception."""
        gemini_mock = AsyncMock(return_value=_err("gemini", "HTTP 429"))
        mistral_mock = AsyncMock(return_value=_err("mistral", "HTTP 503"))
        groq_mock = AsyncMock(return_value=_err("groq", "HTTP 500"))

        with patch.object(planner, "call_gemini", gemini_mock), \
             patch.object(planner, "call_mistral_medium", mistral_mock), \
             patch.object(planner, "call_groq", groq_mock):
            result = _run(planner.plan_task("Some task"))

        self.assertFalse(result["ok"])
        self.assertEqual(result["subtasks"], [])
        self.assertIn("Tous les providers ont échoué", result["error"])
        self.assertIn("gemini", result["error"])
        self.assertIn("mistral", result["error"])
        self.assertIn("groq", result["error"])
        # Les 3 providers doivent tous avoir été tentés.
        gemini_mock.assert_awaited_once()
        mistral_mock.assert_awaited_once()
        groq_mock.assert_awaited_once()
        print("[OK] test_all_providers_fail_returns_error_dict")

    def test_exception_in_provider_does_not_propagate(self):
        """Une exception brute dans un provider est capturée (try/except)."""

        async def _boom(**kwargs):
            raise RuntimeError("network died")

        mistral_mock = AsyncMock(return_value=_ok("mistral_medium"))
        groq_mock = AsyncMock(return_value=_ok("groq"))

        with patch.object(planner, "call_gemini", _boom), \
             patch.object(planner, "call_mistral_medium", mistral_mock), \
             patch.object(planner, "call_groq", groq_mock):
            # Aucune exception ne doit sortir.
            result = _run(planner.plan_task("robust planning"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "mistral_medium")
        mistral_mock.assert_awaited_once()
        groq_mock.assert_not_awaited()
        print("[OK] test_exception_in_provider_does_not_propagate")

    def test_non_quota_error_still_falls_back(self):
        """Une erreur non-429 déclenche aussi le fallback (jamais bloquer)."""
        gemini_mock = AsyncMock(
            return_value=_err("gemini", "HTTP 500: Internal Error")
        )
        mistral_mock = AsyncMock(return_value=_ok("mistral_medium"))

        with patch.object(planner, "call_gemini", gemini_mock), \
             patch.object(planner, "call_mistral_medium", mistral_mock), \
             patch.object(planner, "call_groq", AsyncMock()):
            result = _run(planner.plan_task("task"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "mistral_medium")
        print("[OK] test_non_quota_error_still_falls_back")

    def test_empty_input_rejected_without_calling_llm(self):
        """Input vide → erreur immédiate, aucun provider appelé."""
        gemini_mock = AsyncMock()
        mistral_mock = AsyncMock()
        groq_mock = AsyncMock()

        with patch.object(planner, "call_gemini", gemini_mock), \
             patch.object(planner, "call_mistral_medium", mistral_mock), \
             patch.object(planner, "call_groq", groq_mock):
            result = _run(planner.plan_task("   "))

        self.assertFalse(result["ok"])
        self.assertIn("vide", result["error"])
        gemini_mock.assert_not_awaited()
        mistral_mock.assert_not_awaited()
        groq_mock.assert_not_awaited()
        print("[OK] test_empty_input_rejected_without_calling_llm")


# ---------------------------------------------------------------------------
# Test des logs
# ---------------------------------------------------------------------------

class TestPlannerLogs(unittest.TestCase):
    """Les logs doivent indiquer clairement le fallback déclenché."""

    def test_log_quota_triggers_info_fallback(self):
        """Erreur 429 Gemini → INFO 'quota_exceeded' puis INFO 'after_fallback_count=1'."""
        gemini_mock = AsyncMock(return_value=_err("gemini", "HTTP 429: quota"))
        mistral_mock = AsyncMock(return_value=_ok("mistral_medium"))

        with patch.object(planner, "call_gemini", gemini_mock), \
             patch.object(planner, "call_mistral_medium", mistral_mock), \
             patch.object(planner, "call_groq", AsyncMock()):
            with self.assertLogs("brainflow.planner", level="INFO") as captured:
                _run(planner.plan_task("task"))

        all_msgs = "\n".join(captured.output)
        self.assertIn("provider=gemini quota_exceeded", all_msgs)
        self.assertIn("after_fallback_count=1", all_msgs)
        self.assertIn("provider=mistral_medium", all_msgs)
        print("[OK] test_log_quota_triggers_info_fallback")

    def test_log_all_failed_is_error_level(self):
        """Tous en échec → log ERROR 'planner all_failed'."""
        with patch.object(planner, "call_gemini",
                          AsyncMock(return_value=_err("gemini", "HTTP 429"))), \
             patch.object(planner, "call_mistral_medium",
                          AsyncMock(return_value=_err("mistral", "HTTP 500"))), \
             patch.object(planner, "call_groq",
                          AsyncMock(return_value=_err("groq", "HTTP 500"))):
            with self.assertLogs("brainflow.planner", level="WARNING") as captured:
                _run(planner.plan_task("task"))

        # Au moins un ERROR
        errors = [m for m in captured.output if m.startswith("ERROR:")]
        self.assertGreaterEqual(len(errors), 1)
        self.assertIn("planner all_failed", "\n".join(captured.output))
        print("[OK] test_log_all_failed_is_error_level")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Détecteur de quota ---")
    r1 = unittest.TextTestRunner(verbosity=0).run(
        unittest.TestLoader().loadTestsFromTestCase(TestQuotaDetector)
    )

    print("\n--- Fallback Gemini → Mistral → Groq ---")
    r2 = unittest.TextTestRunner(verbosity=0).run(
        unittest.TestLoader().loadTestsFromTestCase(TestPlannerFallback)
    )

    print("\n--- Logs ---")
    r3 = unittest.TextTestRunner(verbosity=0).run(
        unittest.TestLoader().loadTestsFromTestCase(TestPlannerLogs)
    )

    if r1.wasSuccessful() and r2.wasSuccessful() and r3.wasSuccessful():
        print("\nTous les tests planner_fallback sont passés ✅")
        sys.exit(0)
    else:
        sys.exit(1)
