"""Tests de l'endpoint public `POST /chat`.

Tests unitaires :
- helpers purs (`_step_text`, `_build_final_response`, `_to_agent_steps`)
- route `/chat` via `httpx.AsyncClient` + `ASGITransport`
  (aucun serveur externe requis, `run_workflow` est mocké pour éviter
  les vrais appels LLM et garder les tests rapides et reproductibles).

Exécution :
    python test_chat.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient

# On importe l'app et les helpers APRÈS avoir préparé les mocks.
import main  # noqa: E402
from router import chat as chat_module  # noqa: E402


# ===========================================================================
# 1. Tests unitaires des helpers (purs, pas d'I/O)
# ===========================================================================

def test_step_text_standard_text() -> None:
    assert chat_module._step_text({"text": "Bonjour"}) == "Bonjour"
    assert chat_module._step_text({"text": "  espaces  "}) == "espaces"


def test_step_text_empty_or_invalid() -> None:
    assert chat_module._step_text({}) == ""
    assert chat_module._step_text({"text": ""}) == ""
    assert chat_module._step_text({"text": "   "}) == ""
    assert chat_module._step_text("not-a-dict") == ""  # type: ignore[arg-type]
    assert chat_module._step_text(None) == ""  # type: ignore[arg-type]


def test_step_text_analyse_agent() -> None:
    out = chat_module._step_text({
        "summary": "Résumé court.",
        "observations": ["a", "b"],
        "recommendations": ["r1"],
    })
    assert "Résumé court." in out
    assert "Observations : a | b" in out
    assert "Recommendations : r1" in out


def test_step_text_error_fallback() -> None:
    out = chat_module._step_text({"error": "boom"})
    assert "⚠️" in out and "boom" in out


def test_build_final_response_empty() -> None:
    assert chat_module._build_final_response([]) == ""


def test_build_final_response_single_step() -> None:
    steps = [{
        "id": 1, "description": "desc", "task_type": "simple",
        "agent": {"text": "Réponse unique."},
    }]
    assert chat_module._build_final_response(steps) == "Réponse unique."


def test_build_final_response_multi_step() -> None:
    steps = [
        {"id": 1, "description": "Étape A", "task_type": "simple",
         "agent": {"text": "Texte A"}},
        {"id": 2, "description": "Étape B", "task_type": "analyse",
         "agent": {"text": "Texte B"}},
    ]
    out = chat_module._build_final_response(steps)
    assert "### Étape A" in out
    assert "Texte A" in out
    assert "### Étape B" in out
    assert "Texte B" in out
    assert "---" in out


def test_to_agent_steps_maps_fields() -> None:
    steps = [{
        "id": 1,
        "description": "desc",
        "task_type": "simple",
        "agent": {"ok": True, "text": "hello", "used": "groq"},
        "validation": {"ok": True, "valid": True, "score": 8},
    }]
    result = chat_module._to_agent_steps(steps)
    assert len(result) == 1
    step = result[0]
    assert step.id == 1
    assert step.description == "desc"
    assert step.task_type == "simple"
    assert step.provider == "groq"
    assert step.text == "hello"
    assert step.ok is True
    assert step.valid is True
    assert step.score == 8


# ===========================================================================
# 2. Test de la route `/chat` (mock de `run_workflow`)
# ===========================================================================

_FAKE_WORKFLOW_RESULT: Dict[str, Any] = {
    "ok": True,
    "user_input": "ping",
    "plan": [],
    "steps": [
        {
            "id": 1,
            "description": "Saluer",
            "task_type": "simple",
            "agent": {"ok": True, "text": "Bonjour !", "used": "groq"},
            "validation": {"ok": True, "valid": True, "score": 9},
        }
    ],
    "summary": {"total": 1, "succeeded": 1, "validated": 1, "failed": 0},
}


async def _post(client: AsyncClient, body: Dict[str, Any] | None) -> Any:
    """Helper : POST /chat avec un body JSON (ou vide)."""
    if body is None:
        return await client.post("/chat")
    return await client.post("/chat", json=body)


async def _run_route_tests() -> None:
    transport = ASGITransport(app=main.app)

    async def fake_run_workflow(msg: str) -> Dict[str, Any]:
        assert msg == "ping"
        return _FAKE_WORKFLOW_RESULT

    # On patche `run_workflow` importé dans router.chat
    with patch.object(chat_module, "run_workflow", side_effect=fake_run_workflow):
        async with AsyncClient(transport=transport, base_url="http://test") as client:

            # --- Cas 1 : message valide ---------------------------------
            r = await _post(client, {"message": "ping"})
            assert r.status_code == 200, r.text
            data = r.json()
            assert data["ok"] is True
            assert data["message"] == "ping"
            assert data["response"] == "Bonjour !"
            assert data["summary"] == {
                "total": 1, "succeeded": 1, "validated": 1, "failed": 0,
            }
            assert len(data["agents"]) == 1
            ag = data["agents"][0]
            assert ag["id"] == 1
            assert ag["provider"] == "groq"
            assert ag["text"] == "Bonjour !"
            assert ag["ok"] is True
            assert ag["valid"] is True
            assert ag["score"] == 9
            print("  ✅ POST /chat avec message valide → 200 + structure conforme")

            # --- Cas 2 : message vide → 400 -----------------------------
            r = await _post(client, {"message": ""})
            assert r.status_code == 400
            assert "vide" in r.json()["detail"].lower()
            print("  ✅ POST /chat avec message vide → 400")

            # --- Cas 3 : message blanc → 400 ----------------------------
            r = await _post(client, {"message": "   "})
            assert r.status_code == 400
            print("  ✅ POST /chat avec message blanc → 400")

            # --- Cas 4 : payload invalide (sans `message`) → 422 -------
            r = await _post(client, {})
            assert r.status_code == 422
            print("  ✅ POST /chat sans champ 'message' → 422")

            # --- Cas 5 : payload mal typé → 422 -------------------------
            r = await _post(client, {"message": 123})  # type: ignore[dict-item]
            assert r.status_code == 422
            print("  ✅ POST /chat avec 'message' non-string → 422")


# ===========================================================================
# Runner
# ===========================================================================

def main_runner() -> int:
    print("=" * 66)
    print("  TESTS  —  Endpoint POST /chat")
    print("=" * 66)

    unit_tests: List[tuple[str, Any]] = [
        ("_step_text : texte standard",       test_step_text_standard_text),
        ("_step_text : cas vides/invalides",  test_step_text_empty_or_invalid),
        ("_step_text : analyse_agent",        test_step_text_analyse_agent),
        ("_step_text : fallback error",       test_step_text_error_fallback),
        ("_build_final_response : vide",      test_build_final_response_empty),
        ("_build_final_response : 1 step",    test_build_final_response_single_step),
        ("_build_final_response : N steps",   test_build_final_response_multi_step),
        ("_to_agent_steps : mapping",         test_to_agent_steps_maps_fields),
    ]

    failed = 0
    print("\n[1] Helpers purs")
    for name, func in unit_tests:
        try:
            func()
            print(f"  ✅ {name}")
        except AssertionError as e:
            failed += 1
            print(f"  ❌ {name} — {e}")
        except Exception as e:  # pragma: no cover
            failed += 1
            print(f"  ❌ {name} — {type(e).__name__}: {e}")

    print("\n[2] Route POST /chat (run_workflow mocké)")
    try:
        asyncio.run(_run_route_tests())
    except AssertionError as e:
        failed += 1
        print(f"  ❌ Route test — {e}")
    except Exception as e:  # pragma: no cover
        failed += 1
        print(f"  ❌ Route test — {type(e).__name__}: {e}")

    print("\n" + "=" * 66)
    if failed == 0:
        print(f"  ✅ TOUS LES TESTS PASSENT ({len(unit_tests)} unit + 5 route)")
        print("=" * 66)
        return 0
    print(f"  ❌ {failed} TEST(S) EN ÉCHEC")
    print("=" * 66)
    return 1


if __name__ == "__main__":
    sys.exit(main_runner())
