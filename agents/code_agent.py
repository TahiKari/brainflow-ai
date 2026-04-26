"""Agent Code — gère les tâches de programmation (écriture, refactor, debug).

Logique :
- Tâche "simple"   → Mistral Codestral (modèle spécialisé code)
- Tâche "complexe" → DeepSeek (fort en raisonnement et gros contextes)
- Fallback automatique en cas d'échec sur l'autre provider, puis Groq.

La complexité peut être :
- fournie explicitement via `task["complexity"] in {"simple", "complexe"}`
- détectée automatiquement via des heuristiques simples sur la description.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from services.llm_service import call_codestral, call_deepseek, call_groq


# ---------------------------------------------------------------------------
# Détection de complexité
# ---------------------------------------------------------------------------

# Mots-clés évoquant une tâche code "complexe" (architecture, optimisation...)
_COMPLEXITY_KEYWORDS = {
    "architecture", "architect", "refactor", "refactorer", "refactoring",
    "optimize", "optimise", "optimiser", "optimisation", "performance",
    "scalable", "scalabilité", "design pattern", "patterns",
    "concevoir", "conception", "système", "system",
    "migrer", "migration", "distribué", "distributed",
    "asynchrone", "async", "concurrent", "concurrency",
    "multi-thread", "threading", "parallel", "parallèle",
    "algorithme complexe", "sécurité", "security", "cryptograph",
    "microservice", "monorepo", "ci/cd", "kubernetes", "docker compose",
}

# Seuil de longueur au-dessus duquel on bascule en "complexe".
_COMPLEXITY_LENGTH_THRESHOLD = 300


def _detect_complexity(description: str) -> str:
    """Heuristique simple : retourne 'simple' ou 'complexe'."""
    if not isinstance(description, str):
        return "simple"

    text = description.lower()

    # 1) longueur significative → complexe
    if len(text) >= _COMPLEXITY_LENGTH_THRESHOLD:
        return "complexe"

    # 2) présence d'un mot-clé "complexe"
    for kw in _COMPLEXITY_KEYWORDS:
        if kw in text:
            return "complexe"

    return "simple"


# ---------------------------------------------------------------------------
# Normalisation de l'entrée
# ---------------------------------------------------------------------------

def _normalize_task(task: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extrait (description, complexity) depuis `task`. None si invalide."""
    description: Optional[str] = None
    explicit_complexity: Optional[str] = None

    if isinstance(task, str):
        description = task
    elif isinstance(task, dict):
        # Clés acceptées pour la description
        for key in ("description", "prompt", "task", "content", "text"):
            value = task.get(key)
            if isinstance(value, str) and value.strip():
                description = value
                break

        # Complexité explicite
        raw = task.get("complexity") or task.get("complexité") or task.get("level")
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in ("simple", "complexe", "complex"):
                explicit_complexity = "complexe" if lowered.startswith("complex") else "simple"
    else:
        return None

    if not description or not description.strip():
        return None

    description = description.strip()
    complexity = explicit_complexity or _detect_complexity(description)

    return {
        "description": description,
        "complexity": complexity,
    }


# ---------------------------------------------------------------------------
# Construction de la chaîne de providers selon la complexité
# ---------------------------------------------------------------------------

def _build_chain(complexity: str) -> list[tuple[str, Any]]:
    """Retourne [(name, callable), ...] dans l'ordre primaire → fallbacks."""
    if complexity == "complexe":
        return [
            ("deepseek", call_deepseek),
            ("codestral", call_codestral),
            ("groq", call_groq),
        ]
    # simple (par défaut)
    return [
        ("codestral", call_codestral),
        ("deepseek", call_deepseek),
        ("groq", call_groq),
    ]


# ---------------------------------------------------------------------------
# Fonction publique
# ---------------------------------------------------------------------------

async def handle_code_task(
    task: Union[str, Dict[str, Any]],
    *,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Traite une tâche de code avec routage et fallback automatiques.

    Parameters
    ----------
    task : str | dict
        - str : description brute de la tâche (complexité auto-détectée).
        - dict : {"description": str, "complexity": "simple"|"complexe"} (complexity optionnelle).

    system, temperature, max_tokens, timeout : kwargs transmis au provider.

    Returns
    -------
    dict
        Succès : {"ok": True, "complexity", "used", "text", "response", "attempts"}
        Échec  : {"ok": False, "complexity", "used": None, "text": "", "error", "attempts"}
    """
    normalized = _normalize_task(task)
    if normalized is None:
        return {
            "ok": False,
            "complexity": None,
            "used": None,
            "text": "",
            "error": "Tâche invalide : description vide ou format non reconnu.",
            "attempts": [],
        }

    description = normalized["description"]
    complexity = normalized["complexity"]

    # Kwargs à transmettre au provider (on ne passe que les valeurs fournies).
    kwargs: Dict[str, Any] = {}
    if system is not None:
        kwargs["system"] = system
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout

    chain = _build_chain(complexity)
    attempts: list[Dict[str, Any]] = []

    for name, func in chain:
        try:
            response = await func(description, **kwargs)
        except Exception as exc:  # garde-fou générique
            attempts.append(
                {"provider": name, "ok": False, "error": f"Exception : {exc!s}"}
            )
            continue

        if response.get("ok"):
            attempts.append(
                {
                    "provider": name,
                    "ok": True,
                    "model": response.get("model"),
                }
            )
            return {
                "ok": True,
                "complexity": complexity,
                "used": name,
                "text": response.get("text", ""),
                "response": response,
                "attempts": attempts,
            }

        # Échec "propre" du provider.
        attempts.append(
            {
                "provider": name,
                "ok": False,
                "error": response.get("error", "Erreur inconnue."),
            }
        )

    # Tous les providers ont échoué.
    return {
        "ok": False,
        "complexity": complexity,
        "used": None,
        "text": "",
        "error": "Tous les providers code ont échoué.",
        "attempts": attempts,
    }
