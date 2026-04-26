"""Tests de logs légers pour `router.router` et `services.llm_service`.

Vérifie :
    - provider utilisé (succès)
    - fallback déclenché (échec intermédiaire)
    - erreurs (message tronqué + status_code si HTTP)
    - temps de réponse (duration_ms / attempt_ms / total_ms présents)

On s'appuie sur `unittest.TestCase.assertLogs` (stdlib) qui capture les
LogRecords sans toucher la configuration globale du logger.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import unittest
from typing import Any, Dict
from unittest.mock import patch

import httpx

from router import router as router_module
from services import llm_service


# ---------------------------------------------------------------------------
# Helpers communs
# ---------------------------------------------------------------------------

def _run(coro):
    """Exécute une coroutine, compat Windows + Python 3.11+."""
    return asyncio.get_event_loop().run_until_complete(coro) \
        if sys.version_info < (3, 10) else asyncio.run(coro)


async def _ok_response(prompt, **kwargs):
    return {
        "ok": True,
        "provider": "gemini",
        "model": "gemini-flash-latest",
        "text": "hello",
        "raw": {},
    }


async def _fail_response(prompt, **kwargs):
    return {
        "ok": False,
        "provider": "gemini",
        "error": "forced error for test",
    }


# ---------------------------------------------------------------------------
# 1. Router : log de succès avec provider + total_ms
# ---------------------------------------------------------------------------

class TestRouterLogs(unittest.TestCase):
    """Vérifie les logs émis par `router.router`."""

    def test_log_success_provider_and_total_ms(self):
        """Log INFO 'success used=... fallback_count=0 total_ms=...' au succès direct."""
        with patch.object(router_module, "call_gemini", _ok_response):
            with self.assertLogs("brainflow.router", level="INFO") as captured:
                result = _run(router_module.route_request("simple", "Hello"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["used"], "gemini")

        all_msgs = "\n".join(captured.output)
        # Log d'entrée
        self.assertIn("route_request task_type=simple", all_msgs)
        self.assertIn("codestral=False", all_msgs)
        self.assertIn("chain=gemini,mistral,groq,deepseek,openrouter", all_msgs)
        # Log de succès
        self.assertIn("route_request success used=gemini", all_msgs)
        self.assertIn("fallback_count=0", all_msgs)
        self.assertRegex(all_msgs, r"total_ms=\d+\.\d+")
        self.assertRegex(all_msgs, r"attempt_ms=\d+\.\d+")
        print("[OK] test_log_success_provider_and_total_ms")

    def test_log_fallback_when_first_provider_fails(self):
        """Log WARNING d'échec gemini puis INFO de succès mistral."""
        async def mistral_ok(prompt, **kwargs):
            return {
                "ok": True, "provider": "mistral", "model": "mixtral", "text": "ok", "raw": {},
            }

        with patch.object(router_module, "call_gemini", _fail_response), \
             patch.object(router_module, "call_mistral_by_task", lambda tt, p, **kw: mistral_ok(p)):
            with self.assertLogs("brainflow.router", level="INFO") as captured:
                result = _run(router_module.route_request("simple", "test"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["used"], "mistral")

        all_msgs = "\n".join(captured.output)
        # Log d'échec gemini (fallback déclenché)
        self.assertIn("route_request provider=gemini failed", all_msgs)
        self.assertIn("forced error for test", all_msgs)
        self.assertRegex(all_msgs, r"provider=gemini failed attempt_ms=\d+\.\d+")
        # Log de succès mistral avec fallback_count=1
        self.assertIn("route_request success used=mistral", all_msgs)
        self.assertIn("fallback_count=1", all_msgs)
        print("[OK] test_log_fallback_when_first_provider_fails")

    def test_log_all_failed(self):
        """Log ERROR 'route_request all_failed ...' quand tous les providers échouent."""
        with patch.object(router_module, "call_gemini", _fail_response), \
             patch.object(router_module, "call_mistral_by_task", lambda tt, p, **kw: _fail_response(p)), \
             patch.object(router_module, "call_groq", _fail_response), \
             patch.object(router_module, "call_deepseek", _fail_response), \
             patch.object(router_module, "call_openrouter", _fail_response):
            with self.assertLogs("brainflow.router", level="WARNING") as captured:
                result = _run(router_module.route_request("simple", "test"))

        self.assertFalse(result["ok"])

        all_msgs = "\n".join(captured.output)
        # 5 WARNINGs (un par provider) — format "LEVEL:logger:message"
        warnings = [m for m in captured.output if m.startswith("WARNING:")]
        self.assertGreaterEqual(len(warnings), 5)
        # ERROR final
        self.assertIn("route_request all_failed", all_msgs)
        self.assertIn("providers_tried=gemini,mistral,groq,deepseek,openrouter", all_msgs)
        self.assertRegex(all_msgs, r"total_ms=\d+\.\d+")
        print("[OK] test_log_all_failed")

    def test_log_codestral_triggered_for_code_task(self):
        """Log d'entrée contient codestral=True quand task_type=='code'."""
        async def codestral_ok(prompt, **kwargs):
            return {
                "ok": True, "provider": "codestral", "model": "codestral-latest",
                "text": "def f()", "raw": {},
            }

        with patch.object(router_module, "call_codestral", codestral_ok):
            with self.assertLogs("brainflow.router", level="INFO") as captured:
                result = _run(router_module.route_request("code", "fibonacci"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["used"], "codestral")
        all_msgs = "\n".join(captured.output)
        self.assertIn("codestral=True", all_msgs)
        self.assertIn("chain=codestral,gemini,mistral,groq,deepseek,openrouter", all_msgs)
        self.assertIn("route_request success used=codestral", all_msgs)
        print("[OK] test_log_codestral_triggered_for_code_task")


# ---------------------------------------------------------------------------
# 2. LLM service : log succès + erreurs avec duration_ms
# ---------------------------------------------------------------------------

class _FakeAsyncClient:
    """Faux httpx.AsyncClient pour capture des logs sans I/O réseau."""

    def __init__(self, response_factory):
        self._response_factory = response_factory

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, *args, **kwargs):
        return self._response_factory()


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {
            "choices": [{"message": {"content": "hi"}}]
        }
        self.text = text or ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=self,
            )

    def json(self):
        return self._payload


class TestLLMServiceLogs(unittest.TestCase):
    """Vérifie les logs émis par `services.llm_service`."""

    def setUp(self):
        # Désactive le cache pour éviter de masquer les appels
        self._cache_orig = llm_service.CACHE_ENABLED
        llm_service.CACHE_ENABLED = False

    def tearDown(self):
        llm_service.CACHE_ENABLED = self._cache_orig

    def test_log_success_provider_model_duration(self):
        """Log INFO 'provider=... model=... status=ok duration_ms=...' au succès."""
        def make_client(*args, **kwargs):
            return _FakeAsyncClient(lambda: _FakeResponse(200))

        keys = {"GROQ_API_KEY": "sk-test"}
        with patch.object(llm_service, "load_api_keys", return_value=keys), \
             patch.object(llm_service.httpx, "AsyncClient", make_client):
            with self.assertLogs("brainflow.llm", level="INFO") as captured:
                result = _run(llm_service.call_groq("hello"))

        self.assertTrue(result["ok"])
        all_msgs = "\n".join(captured.output)
        self.assertIn("provider=groq", all_msgs)
        self.assertIn("model=llama-3.1-8b-instant", all_msgs)
        self.assertIn("status=ok", all_msgs)
        self.assertRegex(all_msgs, r"duration_ms=\d+\.\d+")
        print("[OK] test_log_success_provider_model_duration")

    def test_log_http_error_with_status_and_duration(self):
        """Log WARNING avec status=... et duration_ms=... sur HTTP 500."""
        def make_client(*args, **kwargs):
            return _FakeAsyncClient(
                lambda: _FakeResponse(500, text="internal error")
            )

        keys = {"MISTRAL_API_KEY": "sk-m"}
        with patch.object(llm_service, "load_api_keys", return_value=keys), \
             patch.object(llm_service.httpx, "AsyncClient", make_client):
            with self.assertLogs("brainflow.llm", level="WARNING") as captured:
                result = _run(llm_service.call_mistral_simple("hello"))

        self.assertFalse(result["ok"])
        all_msgs = "\n".join(captured.output)
        self.assertIn("provider=mistral", all_msgs)
        self.assertIn("status=500", all_msgs)
        self.assertIn("error=", all_msgs)
        self.assertRegex(all_msgs, r"duration_ms=\d+\.\d+")
        print("[OK] test_log_http_error_with_status_and_duration")

    def test_log_network_error_with_duration(self):
        """Log WARNING 'network_error=...' avec duration_ms sur erreur réseau."""

        class _NetFail:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **kw):
                raise httpx.ConnectError("boom")

        def make_client(*args, **kwargs):
            return _NetFail()

        keys = {"DEEPSEEK_API_KEY": "sk-d"}
        with patch.object(llm_service, "load_api_keys", return_value=keys), \
             patch.object(llm_service.httpx, "AsyncClient", make_client):
            with self.assertLogs("brainflow.llm", level="WARNING") as captured:
                result = _run(llm_service.call_deepseek("hello"))

        self.assertFalse(result["ok"])
        all_msgs = "\n".join(captured.output)
        self.assertIn("provider=deepseek", all_msgs)
        self.assertIn("network_error=", all_msgs)
        self.assertRegex(all_msgs, r"duration_ms=\d+\.\d+")
        print("[OK] test_log_network_error_with_duration")

    def test_log_gemini_success(self):
        """Gemini a son propre chemin → log INFO provider=gemini duration_ms=..."""

        def make_client(*args, **kwargs):
            payload = {
                "candidates": [
                    {"content": {"parts": [{"text": "bonjour"}]}}
                ]
            }
            return _FakeAsyncClient(lambda: _FakeResponse(200, payload=payload))

        keys = {"GEMINI_API_KEY": "g-test"}
        with patch.object(llm_service, "load_api_keys", return_value=keys), \
             patch.object(llm_service.httpx, "AsyncClient", make_client):
            with self.assertLogs("brainflow.llm", level="INFO") as captured:
                result = _run(llm_service.call_gemini("hello", use_cache=False))

        self.assertTrue(result["ok"])
        all_msgs = "\n".join(captured.output)
        self.assertIn("provider=gemini", all_msgs)
        self.assertIn("status=ok", all_msgs)
        self.assertRegex(all_msgs, r"duration_ms=\d+\.\d+")
        print("[OK] test_log_gemini_success")


# ---------------------------------------------------------------------------
# Runner CLI (compat avec le style des autres fichiers de tests)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Router : logs ---")
    router_suite = unittest.TestLoader().loadTestsFromTestCase(TestRouterLogs)
    r1 = unittest.TextTestRunner(verbosity=0).run(router_suite)

    print("\n--- LLM service : logs ---")
    llm_suite = unittest.TestLoader().loadTestsFromTestCase(TestLLMServiceLogs)
    r2 = unittest.TextTestRunner(verbosity=0).run(llm_suite)

    if r1.wasSuccessful() and r2.wasSuccessful():
        print("\nTous les tests de logs sont passés ✅")
        sys.exit(0)
    else:
        sys.exit(1)
