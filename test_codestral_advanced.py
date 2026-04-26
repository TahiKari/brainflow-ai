"""Tests avancés / edge cases pour l'intégration Codestral.

Couverture :
1. Edge cases du détecteur `_should_use_codestral` :
   - prompts accentués (FR)
   - markdown de code (```...```)
   - emojis
   - mots contenant un mot-clé comme sous-chaîne ("pythonic", "encoder")
   - ponctuation variée
   - task_type None / vide / non-chaîne
2. Comportement de `call_codestral` quand `CODESTRAL_API_KEY` est absente :
   - retombe sur `MISTRAL_API_KEY` + endpoint Mistral standard
3. Concurrence : plusieurs `route_request` en parallèle (code + non-code).
4. Intégration avec `code_agent.handle_code_task` (autre point d'entrée
   utilisant Codestral) — non-régression.

Exécution :
    python test_codestral_advanced.py
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import patch

from router import router as router_mod
from router.router import CODESTRAL_TRIGGER_KEYWORDS, _should_use_codestral
from agents import code_agent as code_agent_mod
from services import llm_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(provider: str, model: str, text: str = "ok") -> Dict[str, Any]:
    return {"ok": True, "provider": provider, "model": model, "text": text, "raw": {}}


def _ko(provider: str, error: str = "mocked error") -> Dict[str, Any]:
    return {"ok": False, "provider": provider, "error": error, "quota_exceeded": False}


# ---------------------------------------------------------------------------
# 1. Edge cases du détecteur
# ---------------------------------------------------------------------------

def test_detector_edge_cases() -> None:
    # --- TRUE attendu ---
    # Français accentué
    assert _should_use_codestral(None, "Corrige le bug dans mon script Python") is True
    assert _should_use_codestral(None, "J'ai besoin d'un SCRIPT bash") is True
    # Markdown code block (contient forcément "code" ? non → contient "python" si fence ```python)
    assert _should_use_codestral(None, "Voici mon code :\n```python\nprint('hi')\n```") is True
    # Emojis + mot-clé
    assert _should_use_codestral(None, "🐛 Fix this bug please 🙏") is True
    # Ponctuation collée
    assert _should_use_codestral(None, "Debug: my-code.js ?") is True
    # Matching partiel voulu : "pythonic" contient "python" → True (comportement documenté)
    assert _should_use_codestral(None, "Make it more pythonic") is True, \
        "La détection est une recherche de sous-chaîne : 'pythonic' matche 'python'."
    # task_type insensible casse/espaces
    assert _should_use_codestral("CODE", "") is True
    assert _should_use_codestral("\tcode\n", "") is True

    # --- FALSE attendu ---
    # Aucun mot-clé
    assert _should_use_codestral(None, "Rédige un poème sur la mer") is False
    assert _should_use_codestral("simple", "Quel temps fait-il à Paris ?") is False
    assert _should_use_codestral("analyse", "Analyse ces ventes du Q3") is False
    # task_type proche mais différent
    assert _should_use_codestral("coder", "rédaction") is False
    assert _should_use_codestral("codage", "rédaction") is False
    # Inputs vides / invalides
    assert _should_use_codestral("", "") is False
    assert _should_use_codestral(None, "") is False
    assert _should_use_codestral(None, None) is False  # type: ignore[arg-type]
    # task_type non-chaîne
    assert _should_use_codestral(123, "hello") is False  # type: ignore[arg-type]
    # Prompt non-chaîne
    assert _should_use_codestral(None, ["code"]) is False  # type: ignore[arg-type]

    # Sanity : la liste des mots-clés reste celle attendue
    assert CODESTRAL_TRIGGER_KEYWORDS == ("code", "bug", "python", "script")

    print("[OK] test_detector_edge_cases")


# ---------------------------------------------------------------------------
# 2. call_codestral sans CODESTRAL_API_KEY → fallback MISTRAL_API_KEY
# ---------------------------------------------------------------------------

async def test_call_codestral_without_dedicated_key() -> None:
    """Quand CODESTRAL_API_KEY absente mais MISTRAL_API_KEY présente,
    `call_codestral` doit émettre un appel HTTP avec la clé Mistral et
    l'endpoint adéquat (soit codestral.mistral.ai, soit l'endpoint Mistral
    standard selon l'implémentation)."""

    captured: Dict[str, Any] = {}

    class FakeResponse:
        status_code = 200
        headers: Dict[str, str] = {}

        def __init__(self) -> None:
            self._json = {
                "choices": [
                    {"message": {"content": "print('hello')"}}
                ],
                "model": "codestral-latest",
            }

        def json(self) -> Dict[str, Any]:
            return self._json

        @property
        def text(self) -> str:
            return "ok"

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, url: str, *, headers: Dict[str, str],
                       json: Dict[str, Any]) -> FakeResponse:
            captured["url"] = url
            captured["auth"] = headers.get("Authorization", "")
            captured["payload"] = json
            return FakeResponse()

    # Clés : CODESTRAL_API_KEY absente, MISTRAL_API_KEY présente.
    fake_keys = {"MISTRAL_API_KEY": "mistral-xyz"}

    with patch.object(llm_service, "load_api_keys", return_value=fake_keys), \
         patch.object(llm_service.httpx, "AsyncClient", FakeAsyncClient):
        # Désactive le cache pour ce test
        prev_cache = llm_service.CACHE_ENABLED
        llm_service.CACHE_ENABLED = False
        try:
            resp = await llm_service.call_codestral(
                "write a function",
                temperature=0.1,
            )
        finally:
            llm_service.CACHE_ENABLED = prev_cache

    assert resp.get("ok") is True, f"resp KO : {resp}"
    # La clé Mistral a bien été utilisée en fallback.
    assert "mistral-xyz" in captured.get("auth", ""), captured.get("auth")
    # URL : peu importe laquelle des deux (codestral.mistral.ai OU api.mistral.ai/...).
    # On vérifie juste que c'est bien un endpoint Mistral.
    assert "mistral.ai" in captured.get("url", ""), captured.get("url")
    print("[OK] test_call_codestral_without_dedicated_key "
          f"(url={captured.get('url')})")


async def test_call_codestral_without_any_key() -> None:
    """Sans aucune clé (ni CODESTRAL ni MISTRAL), `call_codestral` doit
    renvoyer `ok=False` avec une erreur explicite et sans lever d'exception."""
    with patch.object(llm_service, "load_api_keys", return_value={}):
        prev_cache = llm_service.CACHE_ENABLED
        llm_service.CACHE_ENABLED = False
        try:
            resp = await llm_service.call_codestral("hello")
        finally:
            llm_service.CACHE_ENABLED = prev_cache

    assert resp.get("ok") is False
    assert resp.get("provider") in ("codestral", "mistral", None, "")
    assert "error" in resp and isinstance(resp["error"], str) and resp["error"]
    print(f"[OK] test_call_codestral_without_any_key (error={resp['error']!r})")


# ---------------------------------------------------------------------------
# 3. Concurrence : plusieurs route_request en parallèle
# ---------------------------------------------------------------------------

async def test_concurrent_routing() -> None:
    """Mix de 6 requêtes en parallèle : 3 code + 3 chat.
    - Les 3 'code' doivent être servies par Codestral.
    - Les 3 'chat' doivent être servies par Gemini.
    - Codestral ne doit JAMAIS être appelé pour un chat.
    """
    codestral_calls: List[str] = []
    gemini_calls: List[str] = []

    lock = asyncio.Lock()

    async def fake_codestral(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        async with lock:
            codestral_calls.append(prompt)
        await asyncio.sleep(0.01)  # simule latence I/O
        return _ok("codestral", "codestral-latest", text=f"[codestral] {prompt[:20]}")

    async def fake_gemini(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        async with lock:
            gemini_calls.append(prompt)
        await asyncio.sleep(0.01)
        return _ok("gemini", "gemini-flash-latest", text=f"[gemini] {prompt[:20]}")

    code_prompts = [
        "Write a Python function",
        "Fix this bug in my code",
        "Refactor this bash script",
    ]
    chat_prompts = [
        "Bonjour, quelle est la météo ?",
        "Raconte-moi une blague",
        "Quelle est la capitale de l'Italie ?",
    ]

    with patch.object(router_mod, "call_codestral", side_effect=fake_codestral), \
         patch.object(router_mod, "call_gemini", side_effect=fake_gemini):
        tasks = (
            [router_mod.route_request("code", p) for p in code_prompts]
            + [router_mod.route_request("simple", p) for p in chat_prompts]
        )
        results = await asyncio.gather(*tasks)

    # Les 3 premiers résultats viennent du code path → Codestral.
    for r, p in zip(results[:3], code_prompts):
        assert r["ok"] is True, f"code req KO : {r}"
        assert r["used"] == "codestral", f"attendu codestral pour {p!r}, got {r['used']}"

    # Les 3 suivants viennent du chat path → Gemini.
    for r, p in zip(results[3:], chat_prompts):
        assert r["ok"] is True, f"chat req KO : {r}"
        assert r["used"] == "gemini", f"attendu gemini pour {p!r}, got {r['used']}"

    assert len(codestral_calls) == 3, f"Codestral appelé {len(codestral_calls)} fois (attendu 3)"
    assert len(gemini_calls) == 3, f"Gemini appelé {len(gemini_calls)} fois (attendu 3)"
    # Aucun prompt "chat" n'a atteint Codestral.
    assert not any(p in codestral_calls for p in chat_prompts)

    print(f"[OK] test_concurrent_routing (codestral={len(codestral_calls)}, "
          f"gemini={len(gemini_calls)})")


# ---------------------------------------------------------------------------
# 4. Non-régression : code_agent.handle_code_task utilise toujours Codestral
# ---------------------------------------------------------------------------

async def test_code_agent_still_uses_codestral_first() -> None:
    """`code_agent.handle_code_task` (autre point d'entrée) doit toujours
    essayer Codestral en premier pour une tâche simple."""
    called: List[str] = []

    async def fake_codestral(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        called.append("codestral")
        return _ok("codestral", "codestral-latest", text="def f(): pass")

    async def fake_deepseek(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        called.append("deepseek")
        return _ok("deepseek", "ds")

    async def fake_groq(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        called.append("groq")
        return _ok("groq", "g")

    with patch.object(code_agent_mod, "call_codestral", side_effect=fake_codestral), \
         patch.object(code_agent_mod, "call_deepseek", side_effect=fake_deepseek), \
         patch.object(code_agent_mod, "call_groq", side_effect=fake_groq):
        result = await code_agent_mod.handle_code_task("Write a hello world function")

    assert result["ok"] is True
    assert result["used"] == "codestral"
    assert called == ["codestral"], f"chaîne inattendue : {called}"
    print("[OK] test_code_agent_still_uses_codestral_first")


# ---------------------------------------------------------------------------
# 5. Intégration via endpoint HTTP /api/chat (chat_routes → workflow → router)
# ---------------------------------------------------------------------------

async def test_http_chat_endpoint_routes_to_codestral() -> None:
    """Appel HTTP complet sur /api/chat avec un prompt code :
    le pipeline complet doit router vers Codestral quelque part dans la chaîne."""
    try:
        import httpx
        from main import app  # FastAPI app
    except Exception as exc:
        print(f"[SKIP] test_http_chat_endpoint_routes_to_codestral ({exc})")
        return

    # On mocke Codestral + Gemini + Mistral + DeepSeek + Groq au niveau
    # llm_service pour éviter tout appel réseau et observer les appels.
    codestral_prompts: List[str] = []

    async def fake_codestral(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        codestral_prompts.append(prompt)
        return _ok("codestral", "codestral-latest", text="def add(a,b): return a+b")

    async def fake_ok(provider: str):
        async def _fn(prompt: str, **kwargs: Any) -> Dict[str, Any]:
            return _ok(provider, f"{provider}-model", text=f"[{provider}] {prompt[:30]}")
        return _fn

    # Planner et validator : on les mocke pour produire un plan code simple.
    # Important : le workflow a importé `plan_task` / `validate_response`
    # au niveau module → on patch sur `services.workflow`, pas sur l'origine.
    from services import workflow as workflow_mod

    async def fake_planner(user_input: str) -> Dict[str, Any]:
        return {
            "ok": True,
            "subtasks": [
                {"id": 1, "description": user_input, "task_type": "code"}
            ],
        }

    async def fake_validator(text: str) -> Dict[str, Any]:
        return {"ok": True, "valid": True, "score": 95, "used": "mock"}

    gem = await fake_ok("gemini")
    ds = await fake_ok("deepseek")
    grq = await fake_ok("groq")

    try:
        transport_cls = httpx.ASGITransport
    except AttributeError:
        print("[SKIP] test_http_chat_endpoint_routes_to_codestral (httpx trop ancien)")
        return

    with patch.object(code_agent_mod, "call_codestral", side_effect=fake_codestral), \
         patch.object(code_agent_mod, "call_deepseek", side_effect=ds), \
         patch.object(code_agent_mod, "call_groq", side_effect=grq), \
         patch.object(router_mod, "call_codestral", side_effect=fake_codestral), \
         patch.object(router_mod, "call_gemini", side_effect=gem), \
         patch.object(workflow_mod, "plan_task", side_effect=fake_planner), \
         patch.object(workflow_mod, "validate_response", side_effect=fake_validator):
        transport = transport_cls(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/chat",
                json={"message": "Écris une fonction python qui additionne deux nombres"},
            )

    assert r.status_code == 200, f"HTTP {r.status_code} : {r.text}"
    data = r.json()
    assert data.get("ok") is True, f"chat KO : {data}"
    # Au moins une sous-tâche code a atteint Codestral.
    assert codestral_prompts, "Codestral n'a pas été appelé via le pipeline HTTP."
    # La réponse finale contient le texte mocké de Codestral.
    assert "def add" in (data.get("response") or ""), data.get("response")
    print(f"[OK] test_http_chat_endpoint_routes_to_codestral "
          f"(codestral_calls={len(codestral_prompts)})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def _run_all() -> None:
    print("--- Edge cases détecteur ---")
    test_detector_edge_cases()

    print("\n--- call_codestral sans clé dédiée ---")
    await test_call_codestral_without_dedicated_key()
    await test_call_codestral_without_any_key()

    print("\n--- Concurrence ---")
    await test_concurrent_routing()

    print("\n--- Non-régression code_agent ---")
    await test_code_agent_still_uses_codestral_first()

    print("\n--- Endpoint HTTP /api/chat ---")
    await test_http_chat_endpoint_routes_to_codestral()

    print("\nTous les tests avancés Codestral sont passés ✅")


if __name__ == "__main__":
    asyncio.run(_run_all())
