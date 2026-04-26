"""Agent Analyse — produit une analyse structurée d'un contenu arbitraire.

IA primaire : Google Gemini (via services.llm_service.call_gemini).
Fallback   : Groq → OpenRouter (providers orientés raisonnement).

Entrée supportée :
- str  : texte / logs / code / CSV / JSON stringifié
- dict : objet structuré (sérialisé en JSON avant analyse)
- list : liste (sérialisée en JSON)

Sortie : dict normalisé avec summary / observations / patterns /
         anomalies / insights / recommendations.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

from services.llm_service import call_gemini, call_groq, call_openrouter


# ---------------------------------------------------------------------------
# Normalisation de l'entrée
# ---------------------------------------------------------------------------

_MAX_INPUT_CHARS = 20_000  # garde-fou anti-débordement de contexte


def _normalize_input(data: Union[str, Dict[str, Any], List[Any], None]) -> Optional[str]:
    """Transforme l'entrée en texte exploitable par le LLM.

    - str    : retournée telle quelle (strippée).
    - dict   : si une clé de contenu évidente existe, on l'utilise ; sinon JSON.
    - list   : sérialisée en JSON.
    - autres : None.
    """
    if data is None:
        return None

    if isinstance(data, str):
        text = data.strip()
        return text or None

    if isinstance(data, dict):
        # Essai d'extraction directe d'un contenu textuel
        for key in ("data", "text", "content", "input", "payload", "body"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        # Sinon, sérialisation JSON complète
        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            return str(data)

    if isinstance(data, list):
        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            return str(data)

    return None


def _extract_depth(data: Any, explicit: Optional[str]) -> str:
    """Retourne 'rapide' ou 'approfondi'. Défaut 'approfondi'."""
    if isinstance(explicit, str):
        low = explicit.strip().lower()
        if low in ("rapide", "quick", "fast", "shallow"):
            return "rapide"
        if low in ("approfondi", "deep", "detailed", "thorough"):
            return "approfondi"

    if isinstance(data, dict):
        raw = data.get("depth") or data.get("profondeur")
        if isinstance(raw, str):
            low = raw.strip().lower()
            if low in ("rapide", "quick", "fast", "shallow"):
                return "rapide"
            if low in ("approfondi", "deep", "detailed", "thorough"):
                return "approfondi"

    return "approfondi"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "Tu es un agent d'analyse de données expert. Tu examines un contenu "
    "fourni (texte, logs, JSON, code, etc.) et tu en produis une analyse "
    "structurée. Tu réponds UNIQUEMENT par un objet JSON strict valide, "
    "sans texte avant/après, sans blocs markdown."
)


def _build_user_prompt(content: str, depth: str) -> str:
    mode = (
        "Analyse APPROFONDIE : sois exhaustif, identifie les relations "
        "subtiles, les causes possibles et propose des actions détaillées."
        if depth == "approfondi"
        else "Analyse RAPIDE : synthèse concise, focus sur l'essentiel."
    )
    return (
        f"{mode}\n\n"
        "Produis STRICTEMENT ce JSON (clés obligatoires, listes possiblement vides) :\n"
        "{\n"
        '  "summary": "résumé synthétique en 1-3 phrases",\n'
        '  "observations": ["faits bruts constatés dans les données"],\n'
        '  "patterns": ["motifs, régularités, tendances"],\n'
        '  "anomalies": ["éléments atypiques, incohérences, alertes"],\n'
        '  "insights": ["enseignements de plus haut niveau"],\n'
        '  "recommendations": ["actions concrètes suggérées"]\n'
        "}\n\n"
        "Contraintes :\n"
        "- Réponds en français.\n"
        "- Chaque élément de liste = phrase courte et actionnable.\n"
        "- Pas de champ supplémentaire, pas de commentaire, pas de markdown.\n\n"
        "=== CONTENU À ANALYSER ===\n"
        f"{content}\n"
        "=== FIN DU CONTENU ==="
    )


# ---------------------------------------------------------------------------
# Parsing de la sortie
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _try_json(raw: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _parse_analysis(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extrait l'objet JSON d'analyse depuis la sortie LLM.

    Tolère : fences markdown, texte parasite, clés manquantes.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    text = raw_text.strip()
    obj = _try_json(text)

    if obj is None:
        stripped = _CODE_FENCE_RE.sub("", text).strip()
        obj = _try_json(stripped)

    if obj is None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            obj = _try_json(match.group(0))

    if obj is None:
        return None

    def _to_str_list(v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    return {
        "summary": str(obj.get("summary", "")).strip(),
        "observations": _to_str_list(obj.get("observations")),
        "patterns": _to_str_list(obj.get("patterns")),
        "anomalies": _to_str_list(obj.get("anomalies")),
        "insights": _to_str_list(obj.get("insights")),
        "recommendations": _to_str_list(obj.get("recommendations")),
    }


# ---------------------------------------------------------------------------
# Chaîne de providers
# ---------------------------------------------------------------------------

def _build_chain() -> List[tuple[str, Any]]:
    """Primaire Gemini, puis fallbacks Groq et OpenRouter."""
    return [
        ("gemini", call_gemini),
        ("groq", call_groq),
        ("openrouter", call_openrouter),
    ]


# ---------------------------------------------------------------------------
# Fonction publique
# ---------------------------------------------------------------------------

async def analyze_data(
    input: Union[str, Dict[str, Any], List[Any], None],
    *,
    depth: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: Optional[int] = 1200,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Analyse un contenu arbitraire via Gemini (+ fallback Groq/OpenRouter).

    Parameters
    ----------
    input : str | dict | list
        Contenu à analyser.
    depth : "rapide" | "approfondi" | None
        Mode d'analyse. Défaut : "approfondi".
    system : str | None
        System prompt personnalisé (sinon prompt interne d'analyse).
    temperature, max_tokens, timeout : kwargs transmis au provider.

    Returns
    -------
    dict
        Schéma : voir docstring du module.
    """
    # 1) Normalisation
    content = _normalize_input(input)
    if not content:
        return {
            "ok": False,
            "depth": None,
            "used": None,
            "summary": "",
            "observations": [],
            "patterns": [],
            "anomalies": [],
            "insights": [],
            "recommendations": [],
            "raw_text": "",
            "error": "Input vide ou format non reconnu.",
            "attempts": [],
        }

    # Tronquage de sécurité
    if len(content) > _MAX_INPUT_CHARS:
        content = content[:_MAX_INPUT_CHARS] + "\n\n[...tronqué...]"

    d = _extract_depth(input, depth)

    # 2) Prompt
    user_prompt = _build_user_prompt(content, d)
    sys_prompt = system if isinstance(system, str) and system.strip() else _SYSTEM_PROMPT

    # 3) Chaîne primaire + fallbacks
    chain = _build_chain()
    attempts: List[Dict[str, Any]] = []

    call_kwargs: Dict[str, Any] = {"system": sys_prompt, "temperature": temperature}
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        call_kwargs["timeout"] = timeout

    for name, func in chain:
        try:
            resp = await func(user_prompt, **call_kwargs)
        except Exception as exc:
            attempts.append({"provider": name, "ok": False,
                             "error": f"Exception : {exc!s}"})
            continue

        if not resp.get("ok"):
            attempts.append({"provider": name, "ok": False,
                             "error": resp.get("error", "Erreur inconnue.")})
            continue

        raw_text = resp.get("text", "")
        parsed = _parse_analysis(raw_text)

        if parsed is None:
            attempts.append({
                "provider": name,
                "ok": False,
                "error": "Sortie non-JSON ou illisible.",
                "raw_text": raw_text[:300],
            })
            continue

        attempts.append({"provider": name, "ok": True,
                         "model": resp.get("model")})
        return {
            "ok": True,
            "depth": d,
            "used": name,
            "summary": parsed["summary"],
            "observations": parsed["observations"],
            "patterns": parsed["patterns"],
            "anomalies": parsed["anomalies"],
            "insights": parsed["insights"],
            "recommendations": parsed["recommendations"],
            "raw_text": raw_text,
            "attempts": attempts,
        }

    # Tous les providers ont échoué
    return {
        "ok": False,
        "depth": d,
        "used": None,
        "summary": "",
        "observations": [],
        "patterns": [],
        "anomalies": [],
        "insights": [],
        "recommendations": [],
        "raw_text": "",
        "error": "Tous les providers d'analyse ont échoué.",
        "attempts": attempts,
    }
