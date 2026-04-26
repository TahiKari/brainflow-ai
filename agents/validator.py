"""Agent Validator — vérifie la qualité/cohérence d'une réponse d'agent.

Logique :
- Réponse "simple"   → validation par Groq (rapide, gratuit).
- Réponse "critique" → validation par OpenRouter (modèle plus puissant).

Si la validation échoue (erreur réseau, timeout, JSON illisible, etc.),
on relance automatiquement (`max_retries`) sur le même provider, puis on
bascule sur les providers de fallback.

La criticité peut être :
- fournie explicitement (`criticality="simple"|"critique"` ou dans le dict),
- détectée automatiquement via heuristiques sur le contenu.

Retour uniforme (dict) : voir `validate_response` pour le schéma.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

from services.llm_service import call_deepseek, call_groq, call_openrouter


# ---------------------------------------------------------------------------
# Détection de criticité
# ---------------------------------------------------------------------------

_CRITICAL_KEYWORDS = {
    "production", "prod", "release", "deploy", "déploiement",
    "critique", "critical", "sécurité", "security", "vulnerab", "vulnérab",
    "finance", "financier", "paiement", "payment", "bancaire", "banking",
    "médical", "medical", "santé", "health", "patient",
    "légal", "legal", "juridique", "contract",
    "auth", "authentif", "password", "mot de passe", "credential",
    "crypto", "chiffrement", "encryption", "token",
    "personal data", "données personnelles", "rgpd", "gdpr",
    "infrastructure", "kubernetes", "database migration",
}

_CRITICAL_LENGTH_THRESHOLD = 500


def _detect_criticality(text: str) -> str:
    """Retourne 'simple' ou 'critique' selon des heuristiques simples."""
    if not isinstance(text, str) or not text.strip():
        return "simple"

    lowered = text.lower()

    if len(lowered) >= _CRITICAL_LENGTH_THRESHOLD:
        return "critique"

    for kw in _CRITICAL_KEYWORDS:
        if kw in lowered:
            return "critique"

    return "simple"


# ---------------------------------------------------------------------------
# Extraction du contenu à valider
# ---------------------------------------------------------------------------

def _extract_text(response: Union[str, Dict[str, Any]]) -> Optional[str]:
    """Extrait le texte à valider depuis une str ou un dict de réponse d'agent."""
    if isinstance(response, str):
        return response.strip() or None

    if not isinstance(response, dict):
        return None

    # Clés directes
    for key in ("text", "content", "result", "output", "answer", "message"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Réponse imbriquée (ex: code_agent renvoie {"response": {"text": ...}})
    nested = response.get("response")
    if isinstance(nested, dict):
        inner = _extract_text(nested)
        if inner:
            return inner

    return None


def _extract_explicit_criticality(
    response: Union[str, Dict[str, Any]],
) -> Optional[str]:
    """Extrait une criticité explicite depuis un dict de réponse."""
    if not isinstance(response, dict):
        return None
    raw = (
        response.get("criticality")
        or response.get("criticité")
        or response.get("level")
    )
    if isinstance(raw, str):
        low = raw.strip().lower()
        if low in ("critique", "critical"):
            return "critique"
        if low == "simple":
            return "simple"
    return None


# ---------------------------------------------------------------------------
# Prompt de validation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "Tu es un agent validateur expert. Tu analyses une réponse produite par "
    "un autre agent et tu évalues sa qualité : clarté, exactitude, "
    "cohérence, complétude, sécurité. Tu réponds UNIQUEMENT par un objet "
    "JSON strict, sans texte avant ni après, sans blocs de code markdown."
)


def _build_user_prompt(content: str, criticality: str) -> str:
    """Construit le prompt utilisateur demandant une validation JSON stricte."""
    severity = (
        "Mode CRITIQUE : sois strict, recherche activement les failles de "
        "sécurité, erreurs logiques, bugs subtils et zones à risque."
        if criticality == "critique"
        else "Mode SIMPLE : évaluation rapide, focus sur les erreurs évidentes."
    )
    return (
        f"{severity}\n\n"
        "Évalue la réponse suivante et renvoie STRICTEMENT ce JSON :\n"
        "{\n"
        '  "valid": true|false,\n'
        '  "score": 0-10,\n'
        '  "issues": ["..."],\n'
        '  "suggestions": ["..."],\n'
        '  "reason": "explication courte"\n'
        "}\n\n"
        "Règles :\n"
        "- valid = true UNIQUEMENT si la réponse est acceptable en l'état.\n"
        "- score ∈ [0..10] (0=inutilisable, 10=parfait).\n"
        "- issues = liste (éventuellement vide) des problèmes détectés.\n"
        "- suggestions = liste (éventuellement vide) d'améliorations concrètes.\n"
        "- reason = justification courte (≤ 40 mots).\n\n"
        "=== RÉPONSE À ÉVALUER ===\n"
        f"{content}\n"
        "=== FIN ==="
    )


# ---------------------------------------------------------------------------
# Parsing de la sortie JSON du validateur
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _try_json(raw: str) -> Optional[Dict[str, Any]]:
    """Tente de parser un JSON strict. Retourne None si échec."""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def _parse_validation(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extrait l'objet JSON de validation depuis la sortie LLM.

    Tolère : blocs ```json ... ```, texte parasite avant/après, clés manquantes.
    Retourne None si rien d'exploitable.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    text = raw_text.strip()

    # 1) Essai direct
    obj = _try_json(text)

    # 2) Retirer les fences markdown
    if obj is None:
        stripped = _CODE_FENCE_RE.sub("", text).strip()
        obj = _try_json(stripped)

    # 3) Extraire le premier bloc {...} équilibré
    if obj is None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            obj = _try_json(match.group(0))

    if obj is None:
        return None

    # Normalisation des champs
    valid = obj.get("valid")
    if isinstance(valid, str):
        valid = valid.strip().lower() in ("true", "1", "yes", "oui", "ok")
    valid = bool(valid) if isinstance(valid, (bool, int)) else False

    score_raw = obj.get("score")
    try:
        score = int(round(float(score_raw))) if score_raw is not None else 0
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(10, score))

    def _to_str_list(v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    return {
        "valid": valid,
        "score": score,
        "issues": _to_str_list(obj.get("issues")),
        "suggestions": _to_str_list(obj.get("suggestions")),
        "reason": str(obj.get("reason", "")).strip(),
    }


# ---------------------------------------------------------------------------
# Chaîne de providers selon la criticité
# ---------------------------------------------------------------------------

def _build_chain(criticality: str) -> List[tuple[str, Any]]:
    """Ordre : primaire → fallbacks. Chaque provider subit les retries."""
    if criticality == "critique":
        return [
            ("openrouter", call_openrouter),
            ("groq", call_groq),
            ("deepseek", call_deepseek),
        ]
    return [
        ("groq", call_groq),
        ("openrouter", call_openrouter),
        ("deepseek", call_deepseek),
    ]


# ---------------------------------------------------------------------------
# Appel d'un provider avec retries
# ---------------------------------------------------------------------------

async def _call_with_retries(
    name: str,
    func: Any,
    prompt: str,
    *,
    system: str,
    max_retries: int,
    temperature: float,
    max_tokens: Optional[int],
    timeout: Optional[float],
) -> Dict[str, Any]:
    """Appelle `func` jusqu'à `max_retries + 1` fois. Retourne le dernier résultat.

    Ajoute dans le résultat une clé `_tries` = nombre total d'essais effectués.
    """
    kwargs: Dict[str, Any] = {"system": system, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout

    last: Dict[str, Any] = {"ok": False, "provider": name, "error": "no attempt"}
    total = max(1, max_retries + 1)

    for i in range(total):
        try:
            resp = await func(prompt, **kwargs)
        except Exception as exc:
            last = {"ok": False, "provider": name, "error": f"Exception : {exc!s}"}
            continue

        last = resp
        if resp.get("ok"):
            resp["_tries"] = i + 1
            return resp

    last["_tries"] = total
    return last


# ---------------------------------------------------------------------------
# Fonction publique
# ---------------------------------------------------------------------------

async def validate_response(
    response: Union[str, Dict[str, Any]],
    *,
    criticality: Optional[str] = None,
    max_retries: int = 2,
    temperature: float = 0.2,
    max_tokens: Optional[int] = 600,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Valide une réponse d'agent via un LLM externe.

    Parameters
    ----------
    response : str | dict
        Contenu à valider. Si dict, on extrait `text`/`content`/`result`/...
    criticality : "simple" | "critique" | None
        Force la criticité. Si None, détection automatique.
    max_retries : int
        Nombre de relances par provider en cas d'erreur. Défaut 2
        (→ 3 essais max par provider avant fallback).
    temperature, max_tokens, timeout : kwargs transmis au provider.

    Returns
    -------
    dict
        Succès :
            {"ok": True, "criticality", "used", "valid", "score",
             "issues", "suggestions", "reason", "raw_text", "attempts"}
        Échec (aucun provider n'a répondu correctement) :
            {"ok": False, "criticality", "used": None, "valid": False,
             "score": 0, "issues": [], "suggestions": [], "reason": "",
             "raw_text": "", "error", "attempts"}
    """
    # 1) Extraction du texte à valider
    content = _extract_text(response)
    if not content:
        return {
            "ok": False,
            "criticality": None,
            "used": None,
            "valid": False,
            "score": 0,
            "issues": [],
            "suggestions": [],
            "reason": "",
            "raw_text": "",
            "error": "Réponse vide ou format non reconnu.",
            "attempts": [],
        }

    # 2) Détermination de la criticité
    explicit = _extract_explicit_criticality(response)
    if isinstance(criticality, str):
        low = criticality.strip().lower()
        if low in ("critique", "critical"):
            crit = "critique"
        elif low == "simple":
            crit = "simple"
        else:
            crit = explicit or _detect_criticality(content)
    else:
        crit = explicit or _detect_criticality(content)

    # 3) Construction du prompt
    user_prompt = _build_user_prompt(content, crit)

    # 4) Appel en chaîne : primaire (+retries) → fallbacks (+retries)
    chain = _build_chain(crit)
    attempts: List[Dict[str, Any]] = []

    for name, func in chain:
        resp = await _call_with_retries(
            name,
            func,
            user_prompt,
            system=_SYSTEM_PROMPT,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        if not resp.get("ok"):
            attempts.append(
                {
                    "provider": name,
                    "ok": False,
                    "tries": resp.get("_tries", max_retries + 1),
                    "error": resp.get("error", "Erreur inconnue."),
                }
            )
            continue

        raw_text = resp.get("text", "")
        parsed = _parse_validation(raw_text)

        if parsed is None:
            # Provider OK mais JSON illisible → on considère ça comme un échec
            # et on continue la chaîne (équivalent à "erreur → relancer ailleurs").
            attempts.append(
                {
                    "provider": name,
                    "ok": False,
                    "tries": resp.get("_tries", 1),
                    "error": "Sortie non-JSON ou illisible.",
                    "raw_text": raw_text[:300],
                }
            )
            continue

        attempts.append(
            {
                "provider": name,
                "ok": True,
                "tries": resp.get("_tries", 1),
                "model": resp.get("model"),
            }
        )
        return {
            "ok": True,
            "criticality": crit,
            "used": name,
            "valid": parsed["valid"],
            "score": parsed["score"],
            "issues": parsed["issues"],
            "suggestions": parsed["suggestions"],
            "reason": parsed["reason"],
            "raw_text": raw_text,
            "attempts": attempts,
        }

    # Tous les providers ont échoué
    return {
        "ok": False,
        "criticality": crit,
        "used": None,
        "valid": False,
        "score": 0,
        "issues": [],
        "suggestions": [],
        "reason": "",
        "raw_text": "",
        "error": "Tous les providers de validation ont échoué.",
        "attempts": attempts,
    }
