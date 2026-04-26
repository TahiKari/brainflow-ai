"""Service d'appel aux différents fournisseurs LLM.

Fournit des fonctions asynchrones unifiées pour :
- Groq
- Google Gemini
- Mistral (ministral-8b, mixtral-8x7b, codestral, mistral-large)
- DeepSeek
- OpenRouter

Les clés API sont chargées dynamiquement via `services.api_key_manager`.
Toutes les fonctions retournent un dict uniforme :

Succès :
    {"ok": True, "provider": str, "model": str, "text": str, "raw": dict}

Erreur :
    {"ok": False, "provider": str, "error": str}

Aucune exception n'est propagée : les erreurs sont capturées et renvoyées
dans le dict de réponse pour simplifier la consommation par les appelants.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import httpx

from memory.cache import get_cache, make_cache_key, set_cache
from services.api_key_manager import load_api_keys


# ---------------------------------------------------------------------------
# Logger dédié (léger, stdlib)
# ---------------------------------------------------------------------------
logger = logging.getLogger("brainflow.llm")


# ---------------------------------------------------------------------------
# Cache LLM (partagé entre tous les providers)
# ---------------------------------------------------------------------------

# Activé globalement. Peut être désactivé par appel via `use_cache=False`
# ou globalement en mettant CACHE_ENABLED = False ci-dessous.
CACHE_ENABLED: bool = True

# Durée de vie des réponses LLM en secondes (1h par défaut).
# Pour du code déterministe (temperature basse), 1h est raisonnable.
LLM_CACHE_TTL: int = 3600


def _llm_cache_key(
    provider: str,
    prompt: str,
    *,
    model: str,
    system: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
) -> str:
    """Construit une clé de cache déterministe pour un appel LLM."""
    return make_cache_key(
        provider,
        prompt,
        model=model,
        system=system or "",
        temperature=round(float(temperature), 4),
        max_tokens=max_tokens if max_tokens is not None else -1,
    )


def _cache_lookup(key: str) -> Optional[Dict[str, Any]]:
    """Récupère une réponse en cache et la marque `from_cache=True`."""
    if not CACHE_ENABLED:
        return None
    cached = get_cache(key)
    if cached is None:
        return None
    # Copie superficielle pour ne pas polluer l'entrée en cache
    response = dict(cached)
    response["from_cache"] = True
    return response


def _cache_store(key: str, response: Dict[str, Any]) -> None:
    """Stocke une réponse en cache si elle est valide (`ok=True`)."""
    if not CACHE_ENABLED:
        return
    if not isinstance(response, dict):
        return
    if not response.get("ok"):
        return
    # On stocke une copie sans le flag `from_cache` éventuel
    to_store = {k: v for k, v in response.items() if k != "from_cache"}
    set_cache(key, to_store, ttl=LLM_CACHE_TTL)


# ---------------------------------------------------------------------------
# Constantes : endpoints & modèles
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 60.0  # secondes
ROUTER_PROVIDER_TIMEOUT = 10.0  # timeout max par provider côté routing

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
# Endpoint dédié Codestral (clé API `codestral.mistral.ai` distincte de `api.mistral.ai`).
CODESTRAL_URL = "https://codestral.mistral.ai/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)

# Modèles par défaut Mistral
MISTRAL_MODELS = {
    "simple": "ministral-8b-latest",
    "medium": "open-mixtral-8x7b",
    "codestral": "codestral-latest",
    "large": "mistral-large-latest",
}


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _get_key(name: str) -> Optional[str]:
    """Lit la clé `name` depuis le stockage JSON. Retourne None si absente/vide."""
    try:
        keys = load_api_keys()
    except Exception:
        return None
    value = keys.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


# Indicateurs de quota / rate-limit (détection insensible à la casse).
_QUOTA_MARKERS = (
    "quota",
    "quota exceeded",
    "rate limit",
    "rate-limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "resource_exhausted",
    "resource exhausted",
    "insufficient_quota",
    "429",
)


def _is_quota_error(message: str, *, status_code: Optional[int] = None) -> bool:
    """Détecte si un message / code HTTP correspond à une erreur de quota.

    Utilisé notamment pour Gemini : en cas de quota dépassé, on préfère
    sauter immédiatement le provider plutôt que de bloquer l'utilisateur.
    """
    if status_code == 429:
        return True
    if not isinstance(message, str):
        return False
    lowered = message.lower()
    return any(marker in lowered for marker in _QUOTA_MARKERS)


def _error(
    provider: str,
    message: str,
    *,
    quota_exceeded: bool = False,
    status_code: Optional[int] = None,
) -> Dict[str, Any]:
    """Format d'erreur uniforme.

    `quota_exceeded` est positionné à True quand l'erreur correspond à un
    429 / rate-limit / quota — auto-détecté si non passé explicitement.
    """
    if not quota_exceeded:
        quota_exceeded = _is_quota_error(message, status_code=status_code)
    payload: Dict[str, Any] = {
        "ok": False,
        "provider": provider,
        "error": message,
        "quota_exceeded": quota_exceeded,
    }
    if status_code is not None:
        payload["status_code"] = status_code
    return payload


def _success(
    provider: str, model: str, text: str, raw: Dict[str, Any]
) -> Dict[str, Any]:
    """Format de succès uniforme."""
    return {
        "ok": True,
        "provider": provider,
        "model": model,
        "text": text,
        "raw": raw,
    }


def _build_messages(prompt: str, system: Optional[str]) -> list[dict]:
    """Construit la liste `messages` au format OpenAI chat-completions."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _extract_openai_text(data: Dict[str, Any]) -> str:
    """Extrait le texte d'une réponse au format OpenAI chat-completions."""
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


async def _post_openai_compatible(
    *,
    provider: str,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Appelle un endpoint au format OpenAI chat-completions.

    Utilisé pour Groq, Mistral, DeepSeek, OpenRouter.
    """
    # --- Lecture du cache ----------------------------------------------------
    cache_key = ""
    if use_cache and CACHE_ENABLED:
        cache_key = _llm_cache_key(
            provider,
            prompt,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        cached = _cache_lookup(cache_key)
        if cached is not None:
            logger.debug("cache hit provider=%s model=%s", provider, model)
            return cached

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": _build_messages(prompt, system),
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if extra_payload:
        payload.update(extra_payload)

    t_start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        duration_ms = (time.perf_counter() - t_start) * 1000.0
        body = ""
        try:
            body = exc.response.text[:500]
        except Exception:
            pass
        logger.warning(
            "provider=%s model=%s status=%s duration_ms=%.1f error=%s",
            provider,
            model,
            exc.response.status_code,
            duration_ms,
            (body or str(exc))[:200],
        )
        return _error(
            provider,
            f"HTTP {exc.response.status_code} : {body or str(exc)}",
            status_code=exc.response.status_code,
        )
    except httpx.RequestError as exc:
        duration_ms = (time.perf_counter() - t_start) * 1000.0
        logger.warning(
            "provider=%s model=%s duration_ms=%.1f network_error=%s",
            provider, model, duration_ms, exc,
        )
        return _error(provider, f"Erreur réseau : {exc!s}")
    except ValueError as exc:  # JSON decoding
        duration_ms = (time.perf_counter() - t_start) * 1000.0
        logger.warning(
            "provider=%s model=%s duration_ms=%.1f json_error=%s",
            provider, model, duration_ms, exc,
        )
        return _error(provider, f"Réponse JSON invalide : {exc!s}")
    except Exception as exc:  # garde-fou générique
        duration_ms = (time.perf_counter() - t_start) * 1000.0
        logger.warning(
            "provider=%s model=%s duration_ms=%.1f unexpected_error=%s",
            provider, model, duration_ms, exc,
        )
        return _error(provider, f"Erreur inattendue : {exc!s}")

    duration_ms = (time.perf_counter() - t_start) * 1000.0
    logger.info(
        "provider=%s model=%s status=ok duration_ms=%.1f",
        provider, model, duration_ms,
    )

    text = _extract_openai_text(data)
    response = _success(provider, model, text, data)

    # --- Écriture dans le cache ---------------------------------------------
    if use_cache and cache_key:
        _cache_store(cache_key, response)

    return response


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

async def call_groq(
    prompt: str,
    model: str = "llama-3.1-8b-instant",
    *,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Appelle l'API Groq (format OpenAI-compatible)."""
    api_key = _get_key("GROQ_API_KEY")
    if not api_key:
        return _error("groq", "Clé GROQ_API_KEY manquante.")

    return await _post_openai_compatible(
        provider="groq",
        url=GROQ_URL,
        api_key=api_key,
        model=model,
        prompt=prompt,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Gemini (format Google, différent d'OpenAI)
# ---------------------------------------------------------------------------

async def call_gemini(
    prompt: str,
    model: str = "gemini-flash-latest",
    *,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Appelle l'API Google Gemini."""
    api_key = _get_key("GEMINI_API_KEY")
    if not api_key:
        return _error("gemini", "Clé GEMINI_API_KEY manquante.")

    # --- Lecture du cache ----------------------------------------------------
    cache_key = ""
    if use_cache and CACHE_ENABLED:
        cache_key = _llm_cache_key(
            "gemini",
            prompt,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        cached = _cache_lookup(cache_key)
        if cached is not None:
            return cached

    url = GEMINI_URL.format(model=model)
    params = {"key": api_key}

    generation_config: Dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    t_start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                params=params,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        duration_ms = (time.perf_counter() - t_start) * 1000.0
        body = ""
        try:
            body = exc.response.text[:500]
        except Exception:
            pass
        logger.warning(
            "provider=gemini model=%s status=%s duration_ms=%.1f error=%s",
            model,
            exc.response.status_code,
            duration_ms,
            (body or str(exc))[:200],
        )
        return _error(
            "gemini",
            f"HTTP {exc.response.status_code} : {body or str(exc)}",
            status_code=exc.response.status_code,
        )
    except httpx.RequestError as exc:
        duration_ms = (time.perf_counter() - t_start) * 1000.0
        logger.warning(
            "provider=gemini model=%s duration_ms=%.1f network_error=%s",
            model, duration_ms, exc,
        )
        return _error("gemini", f"Erreur réseau : {exc!s}")
    except ValueError as exc:
        duration_ms = (time.perf_counter() - t_start) * 1000.0
        logger.warning(
            "provider=gemini model=%s duration_ms=%.1f json_error=%s",
            model, duration_ms, exc,
        )
        return _error("gemini", f"Réponse JSON invalide : {exc!s}")
    except Exception as exc:
        duration_ms = (time.perf_counter() - t_start) * 1000.0
        logger.warning(
            "provider=gemini model=%s duration_ms=%.1f unexpected_error=%s",
            model, duration_ms, exc,
        )
        return _error("gemini", f"Erreur inattendue : {exc!s}")

    duration_ms = (time.perf_counter() - t_start) * 1000.0
    logger.info(
        "provider=gemini model=%s status=ok duration_ms=%.1f",
        model, duration_ms,
    )

    # Extraction du texte (format Google)
    text = ""
    try:
        candidates = data.get("candidates") or []
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
    except Exception:
        text = ""

    response = _success("gemini", model, text, data)

    # --- Écriture dans le cache ---------------------------------------------
    if use_cache and cache_key:
        _cache_store(cache_key, response)

    return response


# ---------------------------------------------------------------------------
# Mistral (plusieurs variantes selon le modèle)
# ---------------------------------------------------------------------------

async def _call_mistral(
    prompt: str,
    model: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Helper commun pour tous les appels Mistral."""
    api_key = _get_key("MISTRAL_API_KEY")
    if not api_key:
        return _error("mistral", "Clé MISTRAL_API_KEY manquante.")

    return await _post_openai_compatible(
        provider="mistral",
        url=MISTRAL_URL,
        api_key=api_key,
        model=model,
        prompt=prompt,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


async def call_mistral_simple(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Mistral — modèle léger `ministral-8b-latest`."""
    return await _call_mistral(
        prompt,
        MISTRAL_MODELS["simple"],
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


async def call_mistral_medium(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Mistral — modèle intermédiaire `open-mixtral-8x7b`."""
    return await _call_mistral(
        prompt,
        MISTRAL_MODELS["medium"],
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


async def call_codestral(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Codestral — modèle spécialisé code.

    Utilise l'endpoint dédié `codestral.mistral.ai` avec la clé dédiée
    `CODESTRAL_API_KEY`. Si celle-ci est absente, on retombe sur
    `MISTRAL_API_KEY` + endpoint Mistral standard pour rester
    rétro-compatible (`codestral-latest` est également accessible via
    l'API Mistral classique).
    """
    codestral_key = _get_key("CODESTRAL_API_KEY")
    if codestral_key:
        return await _post_openai_compatible(
            provider="codestral",
            url=CODESTRAL_URL,
            api_key=codestral_key,
            model=MISTRAL_MODELS["codestral"],
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    # Fallback : pas de clé dédiée → on utilise la clé Mistral standard.
    mistral_key = _get_key("MISTRAL_API_KEY")
    if not mistral_key:
        return _error(
            "codestral",
            "Clé CODESTRAL_API_KEY manquante (et MISTRAL_API_KEY absente).",
        )
    return await _post_openai_compatible(
        provider="codestral",
        url=MISTRAL_URL,
        api_key=mistral_key,
        model=MISTRAL_MODELS["codestral"],
        prompt=prompt,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


async def call_mistral_large(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Mistral — modèle haut de gamme `mistral-large-latest`."""
    return await _call_mistral(
        prompt,
        MISTRAL_MODELS["large"],
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------

async def call_deepseek(
    prompt: str,
    model: str = "deepseek-chat",
    *,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Appelle l'API DeepSeek (format OpenAI-compatible)."""
    api_key = _get_key("DEEPSEEK_API_KEY")
    if not api_key:
        return _error("deepseek", "Clé DEEPSEEK_API_KEY manquante.")

    return await _post_openai_compatible(
        provider="deepseek",
        url=DEEPSEEK_URL,
        api_key=api_key,
        model=model,
        prompt=prompt,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------

async def call_openrouter(
    prompt: str,
    model: str = "openai/gpt-4o-mini",
    *,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
    referer: Optional[str] = None,
    app_title: Optional[str] = "BrainFlow AI",
) -> Dict[str, Any]:
    """Appelle l'API OpenRouter (format OpenAI-compatible).

    OpenRouter recommande d'envoyer `HTTP-Referer` et `X-Title` pour le ranking.
    """
    api_key = _get_key("OPENROUTER_API_KEY")
    if not api_key:
        return _error("openrouter", "Clé OPENROUTER_API_KEY manquante.")

    extra_headers: Dict[str, str] = {}
    if referer:
        extra_headers["HTTP-Referer"] = referer
    if app_title:
        extra_headers["X-Title"] = app_title

    return await _post_openai_compatible(
        provider="openrouter",
        url=OPENROUTER_URL,
        api_key=api_key,
        model=model,
        prompt=prompt,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_headers=extra_headers or None,
    )


# ---------------------------------------------------------------------------
# Registre : permet d'appeler un provider par son nom (utile côté routes)
# ---------------------------------------------------------------------------

PROVIDERS = {
    "groq": call_groq,
    "gemini": call_gemini,
    "mistral_simple": call_mistral_simple,
    "mistral_medium": call_mistral_medium,
    "codestral": call_codestral,
    "mistral_large": call_mistral_large,
    "deepseek": call_deepseek,
    "openrouter": call_openrouter,
}


# ---------------------------------------------------------------------------
# Sélection dynamique d'un modèle Mistral selon le type de tâche
# ---------------------------------------------------------------------------

def select_mistral_model(task_type: Optional[str]) -> str:
    """Retourne l'identifiant de modèle Mistral adapté au type de tâche.

    Règles :
        - "simple"              → `ministral-8b-latest`
        - "moyen" / "medium"    → `open-mixtral-8x7b`
        - "complexe" / "complex" / "large" → `mistral-large-latest`
        - autre / None          → `open-mixtral-8x7b` (compromis par défaut)

    Retour :
        str : identifiant du modèle Mistral (exploitable par `_call_mistral`).
    """
    key = (task_type or "").strip().lower()
    if key == "simple":
        return MISTRAL_MODELS["simple"]
    if key in ("complexe", "complex", "large"):
        return MISTRAL_MODELS["large"]
    # "moyen" / "medium" ou inconnu → défaut sûr
    return MISTRAL_MODELS["medium"]


async def call_mistral_by_task(
    task_type: Optional[str],
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Appelle Mistral en sélectionnant dynamiquement le modèle via `select_mistral_model`.

    Conserve les helpers existants (`call_mistral_simple`, `call_mistral_medium`,
    `call_mistral_large`) pour la rétro-compatibilité.
    """
    model = select_mistral_model(task_type)
    return await _call_mistral(
        prompt,
        model,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
