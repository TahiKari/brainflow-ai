"""Cache mémoire simple pour réduire les appels API LLM.

Ce module fournit un cache clé → valeur en RAM, avec :
- TTL par entrée (expiration automatique)
- Capacité maximale avec éviction FIFO
- Thread-safe (utilisable depuis du code async)
- Aucune dépendance externe (stdlib only)

Utilisation typique
-------------------
>>> from memory.cache import get_cache, set_cache, make_cache_key
>>> key = make_cache_key("groq", "Bonjour", model="llama-3", temperature=0.2)
>>> cached = get_cache(key)
>>> if cached is None:
...     cached = appeler_api(...)
...     set_cache(key, cached, ttl=3600)

Les fonctions principales demandées sont `get_cache` et `set_cache`.
Les autres (`make_cache_key`, `clear_cache`, `cache_stats`) sont des utilitaires.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_TTL: int = 3600          # 1 heure par défaut
MAX_ENTRIES: int = 500           # capacité max du cache (éviction FIFO)

# ---------------------------------------------------------------------------
# État interne
# ---------------------------------------------------------------------------
# _cache : clé -> (value, expires_at)   expires_at=None => jamais expiré
# OrderedDict pour FIFO d'éviction (préservation de l'ordre d'insertion).
_cache: "OrderedDict[str, tuple[Any, Optional[float]]]" = OrderedDict()
_lock = threading.Lock()
_stats = {"hits": 0, "misses": 0, "evictions": 0, "expired": 0}


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------
def _now() -> float:
    return time.time()


def _is_expired(expires_at: Optional[float]) -> bool:
    return expires_at is not None and _now() >= expires_at


def _evict_if_needed() -> None:
    """Supprime les plus anciennes entrées si on dépasse MAX_ENTRIES."""
    while len(_cache) > MAX_ENTRIES:
        _cache.popitem(last=False)  # FIFO : enlève le plus ancien
        _stats["evictions"] += 1


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------
def get_cache(key: str) -> Optional[Any]:
    """Retourne la valeur mise en cache pour `key`, ou `None` si absente/expirée.

    - Si l'entrée est expirée, elle est supprimée et `None` est retourné.
    - Incrémente les compteurs hits/misses.
    """
    if not isinstance(key, str) or not key:
        return None

    with _lock:
        entry = _cache.get(key)
        if entry is None:
            _stats["misses"] += 1
            return None

        value, expires_at = entry
        if _is_expired(expires_at):
            # Expirée → on la retire
            _cache.pop(key, None)
            _stats["expired"] += 1
            _stats["misses"] += 1
            return None

        _stats["hits"] += 1
        return value


def set_cache(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """Stocke `value` sous `key`.

    - `ttl` : durée de vie en secondes. Si `None`, utilise `DEFAULT_TTL`.
              Si `<= 0`, l'entrée ne sera jamais rendue par `get_cache`.
    - Si la clé existe déjà, elle est mise à jour et replacée en fin de file.
    - Si la capacité est dépassée, la plus ancienne entrée est évincée.
    """
    if not isinstance(key, str) or not key:
        return

    duration = DEFAULT_TTL if ttl is None else int(ttl)
    expires_at: Optional[float]
    if duration <= 0:
        # TTL nul/négatif → on stocke mais ce sera immédiatement expiré.
        expires_at = _now() - 1
    else:
        expires_at = _now() + duration

    with _lock:
        if key in _cache:
            _cache.pop(key)  # on va le réinsérer à la fin
        _cache[key] = (value, expires_at)
        _evict_if_needed()


def make_cache_key(provider: str, prompt: str, **params: Any) -> str:
    """Génère une clé déterministe pour un appel LLM.

    Hash SHA-256 de (provider, prompt, params triés). Deux appels identiques
    produisent la même clé, indépendamment de l'ordre des kwargs.
    """
    provider = str(provider or "")
    prompt = str(prompt or "")
    try:
        # sort_keys pour que l'ordre des params ne change pas le hash.
        params_blob = json.dumps(params, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        params_blob = repr(sorted(params.items()))

    raw = f"{provider}\x1f{prompt}\x1f{params_blob}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def clear_cache() -> int:
    """Vide entièrement le cache. Retourne le nombre d'entrées supprimées."""
    with _lock:
        n = len(_cache)
        _cache.clear()
        # on ne reset pas les stats cumulatives ; utiliser reset_stats() si besoin
    return n


def cache_stats() -> dict:
    """Retourne un snapshot des compteurs et de la taille actuelle."""
    with _lock:
        return {
            "size": len(_cache),
            "max_entries": MAX_ENTRIES,
            "default_ttl": DEFAULT_TTL,
            "hits": _stats["hits"],
            "misses": _stats["misses"],
            "evictions": _stats["evictions"],
            "expired": _stats["expired"],
        }


def reset_stats() -> None:
    """Remet les compteurs de stats à zéro (n'affecte pas les entrées)."""
    with _lock:
        for k in _stats:
            _stats[k] = 0
