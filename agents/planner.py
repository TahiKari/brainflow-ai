"""Agent Planner — décompose une requête utilisateur en sous-tâches.

Essaie Gemini en priorité, puis bascule automatiquement sur Mistral (modèle
moyen) puis Groq en cas d'erreur (notamment HTTP 429 quota dépassé).
Le planner ne dépend donc jamais uniquement de Gemini.

Le résultat est une liste de sous-tâches directement consommables par
`router.router.route_request`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from services.llm_service import call_gemini, call_groq, call_mistral_medium


logger = logging.getLogger("brainflow.planner")


# Les valeurs acceptées pour `task_type` (alignées sur `router.router.TASK_ROUTES`).
VALID_TASK_TYPES = {"simple", "analyse", "code", "moyen", "complexe"}


# ---------------------------------------------------------------------------
# Prompt système : demande une sortie JSON stricte
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """Tu es un planificateur de tâches.

Ton rôle : découper une requête utilisateur en une liste ordonnée de
sous-tâches concrètes, réalisables, et indépendantes.

Règles strictes :
1. Réponds UNIQUEMENT avec un JSON valide. Aucun texte avant ou après.
2. Le JSON doit être une liste d'objets au format :
   [
     {"id": 1, "description": "...", "task_type": "simple"},
     {"id": 2, "description": "...", "task_type": "code"}
   ]
3. `task_type` doit être exactement l'une de ces valeurs :
   - "simple"   (tâche rapide / formulation courte)
   - "analyse"  (analyse, synthèse, réflexion)
   - "code"     (écriture ou explication de code)
   - "moyen"    (raisonnement intermédiaire, plusieurs étapes)
   - "complexe" (raisonnement avancé, tâche difficile)
4. Entre 1 et 8 sous-tâches. Commence par les prérequis.
5. Les `id` commencent à 1 et sont croissants.
"""


# ---------------------------------------------------------------------------
# Helpers de parsing
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """Supprime les éventuels blocs ```json ... ``` autour du texte."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # retire la première ligne (```json ou ```) et la dernière (```).
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _extract_json_array(text: str) -> str | None:
    """Tente de localiser la première liste JSON `[...]` dans le texte."""
    # Recherche greedy sur la première `[` jusqu'à la dernière `]`.
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    return match.group(0) if match else None


def _normalize_subtask(item: Any, fallback_id: int) -> Dict[str, Any] | None:
    """Valide / normalise une sous-tâche issue du JSON."""
    if not isinstance(item, dict):
        return None

    description = item.get("description") or item.get("task") or item.get("desc")
    if not isinstance(description, str) or not description.strip():
        return None

    task_type = item.get("task_type") or item.get("type") or "moyen"
    if not isinstance(task_type, str):
        task_type = "moyen"
    task_type = task_type.strip().lower()
    if task_type not in VALID_TASK_TYPES:
        task_type = "moyen"

    raw_id = item.get("id", fallback_id)
    try:
        sid = int(raw_id)
    except (TypeError, ValueError):
        sid = fallback_id

    return {
        "id": sid,
        "description": description.strip(),
        "task_type": task_type,
    }


def _parse_json_list(text: str) -> List[Dict[str, Any]] | None:
    """Essaie de parser `text` comme une liste JSON de sous-tâches."""
    candidates = [text, _strip_code_fences(text)]
    extracted = _extract_json_array(text)
    if extracted:
        candidates.append(extracted)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (ValueError, json.JSONDecodeError):
            continue

        # Parfois le LLM wrap dans {"subtasks": [...]}
        if isinstance(data, dict):
            for key in ("subtasks", "tasks", "plan", "steps"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break

        if not isinstance(data, list):
            continue

        subtasks: List[Dict[str, Any]] = []
        for idx, item in enumerate(data, start=1):
            normalized = _normalize_subtask(item, fallback_id=idx)
            if normalized is not None:
                subtasks.append(normalized)
        if subtasks:
            return subtasks

    return None


def _fallback_parse_lines(text: str) -> List[Dict[str, Any]]:
    """Dernier recours : extraction des sous-tâches depuis des puces / numéros."""
    subtasks: List[Dict[str, Any]] = []
    line_pattern = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$")
    for line in text.splitlines():
        match = line_pattern.match(line)
        if not match:
            continue
        description = match.group(1).strip()
        if not description:
            continue
        subtasks.append(
            {
                "id": len(subtasks) + 1,
                "description": description,
                "task_type": "moyen",
            }
        )
    return subtasks


# ---------------------------------------------------------------------------
# Chaîne de fallback LLM
# ---------------------------------------------------------------------------

# Ordre imposé par la spec : Gemini → Mistral (moyen) → Groq.
# On stocke des NOMS d'attributs résolus dynamiquement sur ce module à
# chaque appel : cela permet à `unittest.mock.patch.object(planner, "call_x")`
# de fonctionner correctement dans les tests.
_PLANNER_PROVIDERS = (
    ("gemini", "call_gemini"),
    ("mistral_medium", "call_mistral_medium"),
    ("groq", "call_groq"),
)


def _is_quota_error(error: str) -> bool:
    """Détecte une erreur de quota/rate-limit (HTTP 429 typiquement)."""
    if not isinstance(error, str):
        return False
    low = error.lower()
    return (
        "429" in low
        or "quota" in low
        or "rate limit" in low
        or "rate_limit" in low
        or "too many requests" in low
    )


async def _call_planner_llm(prompt: str) -> Dict[str, Any]:
    """Appelle les providers dans l'ordre Gemini → Mistral → Groq.

    Chaque appel est protégé par try/except : aucune exception ne remonte.
    Retourne la première réponse `ok=True`, sinon un dict d'échec listant
    toutes les erreurs rencontrées.
    """
    import sys as _sys

    this_module = _sys.modules[__name__]
    attempts: List[Dict[str, str]] = []

    for name, attr in _PLANNER_PROVIDERS:
        fn = getattr(this_module, attr, None)
        if fn is None:
            attempts.append({"provider": name, "error": f"fonction {attr} indisponible"})
            continue
        try:
            response = await fn(
                prompt=prompt,
                system=_PLANNER_SYSTEM,
                temperature=0.2,
            )
        except Exception as exc:  # défensif : aucune remontée d'exception
            err_msg = f"exception: {exc.__class__.__name__}: {exc}"
            logger.warning("planner provider=%s failed error=%s", name, err_msg[:200])
            attempts.append({"provider": name, "error": err_msg})
            continue

        if response.get("ok"):
            if attempts:
                logger.info(
                    "planner success provider=%s after_fallback_count=%d",
                    name,
                    len(attempts),
                )
            else:
                logger.info("planner success provider=%s", name)
            return {"ok": True, "provider": name, "response": response}

        err_msg = response.get("error") or "erreur inconnue"
        level = "WARNING"
        if name == "gemini" and _is_quota_error(err_msg):
            level = "INFO"  # quota Gemini = cas nominal de fallback
            logger.info(
                "planner provider=gemini quota_exceeded → fallback (error=%s)",
                err_msg[:200],
            )
        else:
            logger.warning(
                "planner provider=%s failed error=%s", name, err_msg[:200]
            )
        attempts.append({"provider": name, "error": err_msg})

    return {"ok": False, "provider": None, "attempts": attempts}


# ---------------------------------------------------------------------------
# Fonction publique
# ---------------------------------------------------------------------------

async def plan_task(user_input: str) -> Dict[str, Any]:
    """Décompose `user_input` en sous-tâches via une chaîne LLM robuste.

    Chaîne : Gemini → Mistral (moyen) → Groq. Le planner ne dépend jamais
    uniquement de Gemini : une erreur HTTP 429 (quota) ou toute autre
    erreur déclenche un fallback automatique.

    Parameters
    ----------
    user_input : str
        La requête utilisateur à planifier.

    Returns
    -------
    dict
        - Succès : {"ok": True, "user_input", "subtasks": [...],
                    "provider": str, "raw": str}
        - Erreur : {"ok": False, "user_input", "subtasks": [],
                    "error": str, "raw": str}

        Chaque sous-tâche a la forme :
            {"id": int, "description": str, "task_type": str}
        où `task_type` ∈ {simple, analyse, code, moyen, complexe}.
    """
    if not isinstance(user_input, str) or not user_input.strip():
        return {
            "ok": False,
            "user_input": user_input or "",
            "subtasks": [],
            "error": "user_input vide ou invalide.",
            "raw": "",
        }

    # Appel LLM avec fallback Gemini → Mistral → Groq.
    llm_result = await _call_planner_llm(user_input.strip())

    if not llm_result.get("ok"):
        attempts = llm_result.get("attempts") or []
        details = " | ".join(
            f"{a['provider']}: {a['error']}" for a in attempts
        ) or "aucun provider disponible"
        logger.error("planner all_failed details=%s", details[:400])
        return {
            "ok": False,
            "user_input": user_input,
            "subtasks": [],
            "error": f"Tous les providers ont échoué ({details})",
            "raw": "",
        }

    provider_used: str = llm_result["provider"]
    response: Dict[str, Any] = llm_result["response"]
    raw_text = response.get("text") or ""

    # 1) Tentative de parsing JSON strict.
    subtasks = _parse_json_list(raw_text)

    # 2) Fallback : extraction ligne-à-ligne si parsing JSON impossible.
    if not subtasks:
        subtasks = _fallback_parse_lines(raw_text)

    if not subtasks:
        return {
            "ok": False,
            "user_input": user_input,
            "subtasks": [],
            "error": "Impossible d'extraire une liste de sous-tâches.",
            "raw": raw_text,
            "provider": provider_used,
        }

    # Renumérotation défensive : ids 1..N contigus.
    for i, st in enumerate(subtasks, start=1):
        st["id"] = i

    return {
        "ok": True,
        "user_input": user_input,
        "subtasks": subtasks,
        "raw": raw_text,
        "provider": provider_used,
    }
