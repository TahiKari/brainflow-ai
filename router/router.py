"""Routeur logique de tâches LLM.

Dispatch un `prompt` vers le provider approprié selon le `task_type`,
avec une chaîne de fallback si le provider principal échoue.

Règles de routage
-----------------
- simple   → Mistral ministral-8b  (via `call_mistral_by_task`)
- moyen    → Mistral Mixtral-8x7b  (via `call_mistral_by_task`)
- complexe → Mistral Large         (via `call_mistral_by_task`)
- code     → Codestral en tête     (voir règle ci-dessous)

Règle Codestral (utilisation intelligente)
------------------------------------------
Codestral est utilisé **uniquement** pour des tâches de code / debug /
scripts. Détection :
    - `task_type == "code"`, OU
    - le prompt contient (insensible à la casse) l'un des mots-clés :
      "code", "bug", "python", "script".

Quand la règle est vraie, Codestral est placé en **tête** de la chaîne
de fallback ; sinon, la chaîne de fallback reste inchangée (aucun
appel à Codestral pour un chat normal, du contenu ou des questions
simples).

Fallback (ordre strict)
-----------------------
Gemini → Mistral → Groq → DeepSeek → OpenRouter

Si tous les providers échouent, la fonction renvoie `ok=False` avec
l'historique complet des tentatives.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from services.llm_service import (
    ROUTER_PROVIDER_TIMEOUT,
    call_codestral,
    call_deepseek,
    call_gemini,
    call_groq,
    call_mistral_by_task,
    call_openrouter,
    select_mistral_model,
)


# ---------------------------------------------------------------------------
# Logger dédié (léger, stdlib)
# ---------------------------------------------------------------------------
logger = logging.getLogger("brainflow.router")


# ---------------------------------------------------------------------------
# Configuration de routage
# ---------------------------------------------------------------------------

# Type alias : une fonction `async` prenant un prompt + kwargs, renvoyant un dict.
LLMCall = Callable[..., Awaitable[Dict[str, Any]]]

# Fallback global imposé (ordre strict).
# 1. Gemini 2. Mistral 3. Groq 4. DeepSeek 5. OpenRouter
FALLBACK_CHAIN: List[str] = [
    "gemini",
    "mistral",
    "groq",
    "deepseek",
    "openrouter",
]

# Mots-clés déclenchant l'usage de Codestral (insensible à la casse).
CODESTRAL_TRIGGER_KEYWORDS: tuple[str, ...] = ("code", "bug", "python", "script")


def _should_use_codestral(task_type: Optional[str], prompt: str) -> bool:
    """Détermine si Codestral doit être utilisé pour cette requête.

    Règles :
        - `task_type == "code"` (insensible à la casse, espaces ignorés), OU
        - le prompt contient l'un des mots-clés `CODESTRAL_TRIGGER_KEYWORDS`
          (recherche insensible à la casse).

    Toute autre requête (chat, contenu, questions simples) → False.
    """
    if isinstance(task_type, str) and task_type.strip().lower() == "code":
        return True
    if not isinstance(prompt, str) or not prompt:
        return False
    lowered = prompt.lower()
    return any(kw in lowered for kw in CODESTRAL_TRIGGER_KEYWORDS)


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _provider_label(provider: str) -> str:
    """Retourne le libellé provider."""
    return provider


def _record_attempt(
    attempts: List[Dict[str, Any]], provider: str, response: Dict[str, Any]
) -> None:
    """Ajoute une entrée à l'historique des tentatives."""
    attempt: Dict[str, Any] = {
        "provider": _provider_label(provider),
        "ok": bool(response.get("ok")),
    }
    if not response.get("ok"):
        attempt["error"] = response.get("error") or "Erreur inconnue"
        if "quota_exceeded" in response:
            attempt["quota_exceeded"] = bool(response.get("quota_exceeded"))
        if "status_code" in response:
            attempt["status_code"] = response.get("status_code")
    else:
        attempt["model"] = response.get("model")
    attempts.append(attempt)


def _final_success(
    task_type: str,
    used: str,
    response: Dict[str, Any],
    attempts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Format de retour en cas de succès."""
    return {
        "ok": True,
        "task_type": task_type,
        "used": _provider_label(used),
        "text": response.get("text", ""),
        "response": response,
        "attempts": attempts,
    }


def _final_failure(
    task_type: str, attempts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Format de retour quand tous les providers ont échoué."""
    return {
        "ok": False,
        "task_type": task_type,
        "used": None,
        "text": "",
        "error": "Tous les providers ont échoué.",
        "attempts": attempts,
    }


# ---------------------------------------------------------------------------
# Fonction publique
# ---------------------------------------------------------------------------

async def _call_with_timeout(func: LLMCall, prompt: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Appelle un provider avec timeout dur pour ne jamais bloquer la réponse."""
    try:
        return await asyncio.wait_for(func(prompt, **kwargs), timeout=ROUTER_PROVIDER_TIMEOUT)
    except asyncio.TimeoutError:
        return {
            "ok": False,
            "error": f"Timeout provider après {ROUTER_PROVIDER_TIMEOUT:.1f}s",
            "status_code": 408,
            "timeout": True,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Exception inattendue : {exc!s}",
        }


def _build_provider_callable(provider: str, task_type: str) -> Optional[LLMCall]:
    """Résout le callable provider selon la chaîne globale.

    Pour Mistral, on utilise `call_mistral_by_task` qui sélectionne dynamiquement
    le modèle via `select_mistral_model(task_type)` :
        - simple   → ministral-8b
        - moyen    → mixtral-8x7b
        - complexe → mistral-large
    """
    if provider == "codestral":
        return call_codestral
    if provider == "gemini":
        return call_gemini
    if provider == "mistral":
        # Wrapper async compatible avec la signature LLMCall(prompt, **kwargs).
        async def _mistral_call(prompt: str, **kwargs: Any) -> Dict[str, Any]:
            return await call_mistral_by_task(task_type, prompt, **kwargs)

        # Pour un debug éventuel : on expose le modèle choisi.
        _mistral_call.__name__ = (
            f"call_mistral_by_task[{select_mistral_model(task_type)}]"
        )
        return _mistral_call
    if provider == "groq":
        return call_groq
    if provider == "deepseek":
        return call_deepseek
    if provider == "openrouter":
        return call_openrouter
    return None


async def route_request(
    task_type: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Routage global provider avec fallback strict et non bloquant."""
    # Construction des kwargs optionnels (on ne passe que ce qui est explicite).
    call_kwargs: Dict[str, Any] = {}
    if system is not None:
        call_kwargs["system"] = system
    if temperature is not None:
        call_kwargs["temperature"] = temperature
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens

    # Contrainte : timeout max 10s/provider côté routeur (timeout externe ignoré).
    call_kwargs["timeout"] = ROUTER_PROVIDER_TIMEOUT

    # Chaîne dynamique : Codestral en tête uniquement pour les tâches code.
    codestral_used = _should_use_codestral(task_type, prompt)
    if codestral_used:
        chain: List[str] = ["codestral", *FALLBACK_CHAIN]
    else:
        chain = list(FALLBACK_CHAIN)

    logger.info(
        "route_request task_type=%s codestral=%s chain=%s",
        task_type, codestral_used, ",".join(chain),
    )

    attempts: List[Dict[str, Any]] = []
    disabled_for_request: set[str] = set()
    t_total = time.perf_counter()

    for idx, provider in enumerate(chain):
        if provider in disabled_for_request:
            continue

        func = _build_provider_callable(provider, task_type)
        if func is None:
            logger.warning("provider=%s unsupported_by_router", provider)
            attempts.append(
                {
                    "provider": provider,
                    "ok": False,
                    "error": "Provider non supporté par le routeur.",
                }
            )
            continue

        t_attempt = time.perf_counter()
        response = await _call_with_timeout(func, prompt, call_kwargs)
        attempt_ms = (time.perf_counter() - t_attempt) * 1000.0
        response.setdefault("provider", provider)
        _record_attempt(attempts, provider, response)

        if response.get("ok"):
            total_ms = (time.perf_counter() - t_total) * 1000.0
            fallback_count = len(attempts) - 1  # nb de providers échoués avant succès
            logger.info(
                "route_request success used=%s fallback_count=%d "
                "attempt_ms=%.1f total_ms=%.1f",
                provider, fallback_count, attempt_ms, total_ms,
            )
            return _final_success(task_type, provider, response, attempts)

        # Échec → on log et on tente le suivant (fallback).
        err = response.get("error") or "Erreur inconnue"
        logger.warning(
            "route_request provider=%s failed attempt_ms=%.1f error=%s",
            provider, attempt_ms, str(err)[:200],
        )

        # Règle métier : si Gemini échoue, ne plus l'utiliser dans CETTE requête.
        if provider == "gemini":
            disabled_for_request.add("gemini")

    total_ms = (time.perf_counter() - t_total) * 1000.0
    logger.error(
        "route_request all_failed task_type=%s total_ms=%.1f providers_tried=%s",
        task_type, total_ms,
        ",".join(a.get("provider", "?") for a in attempts),
    )
    return _final_failure(task_type, attempts)
