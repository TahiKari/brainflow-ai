"""Workflow — orchestre le pipeline complet Planner → Router → Agent → Validator.

Simple et lisible : aucune logique IA ici, juste de l'orchestration.

Pipeline
--------
1. Planner   : `plan_task(user_input)` découpe la requête en sous-tâches.
2. Router    : selon `task_type`, on choisit l'agent :
                 - "code"    → agents.code_agent.handle_code_task
                 - "analyse" → agents.analyse_agent.analyze_data
                 - autres    → router.router.route_request (simple/moyen/complexe)
3. Agent     : exécute la sous-tâche.
4. Validator : `validate_response(agent_output)` évalue la qualité.

Chaque étape est loguée via `logging` (niveau INFO).
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List

from agents.analyse_agent import analyze_data
from agents.code_agent import handle_code_task
from agents.planner import plan_task
from agents.validator import validate_response
from router.router import route_request


# ---------------------------------------------------------------------------
# Détection d'intention (règle simple, mots-clés)
# ---------------------------------------------------------------------------

# Mots-clés déclenchant une intention "direct" (génération / action).
_DIRECT_KEYWORDS = ("répond", "écris", "génère", "fais")

# Mots-clés déclenchant une intention "analysis" (compréhension / raisonnement).
_ANALYSIS_KEYWORDS = ("explique", "analyse", "pourquoi", "comment")


def detect_intent(message: str) -> str:
    """Détermine l'intention utilisateur à partir du message.

    Règles :
        - contient "répond" / "écris" / "génère" / "fais"   → "direct"
        - contient "explique" / "analyse" / "pourquoi" / "comment" → "analysis"
        - sinon                                              → "direct"

    Les mots-clés "direct" sont prioritaires si les deux familles apparaissent.
    La comparaison est insensible à la casse.
    """
    if not isinstance(message, str):
        return "direct"

    text = message.lower()

    if any(kw in text for kw in _DIRECT_KEYWORDS):
        return "direct"
    if any(kw in text for kw in _ANALYSIS_KEYWORDS):
        return "analysis"
    return "direct"


# ---------------------------------------------------------------------------
# Logger dédié
# ---------------------------------------------------------------------------

logger = logging.getLogger("brainflow.workflow")

if not logger.handlers:
    # Configuration minimaliste (console). N'affecte pas les autres loggers.
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] %(levelname)s | %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Dispatch : task_type → agent
# ---------------------------------------------------------------------------

async def _run_code(description: str) -> Dict[str, Any]:
    """Exécute une sous-tâche de type 'code'."""
    return await handle_code_task(description)


async def _run_analyse(description: str) -> Dict[str, Any]:
    """Exécute une sous-tâche de type 'analyse'."""
    return await analyze_data(description)


async def _run_via_router(task_type: str, description: str) -> Dict[str, Any]:
    """Exécute une sous-tâche 'simple' / 'moyen' / 'complexe' via le router."""
    return await route_request(task_type, description)


# ---------------------------------------------------------------------------
# Mode "direct" : un seul appel LLM, pas de planner ni de validator.
# ---------------------------------------------------------------------------

async def _run_direct(user_input: str) -> Dict[str, Any]:
    """Exécute un appel LLM unique et retourne directement le texte.

    Utilisé quand `detect_intent(user_input) == "direct"` :
    l'utilisateur attend une réponse immédiate, pas une décomposition.

    Conserve le même schéma de retour que `run_workflow` (clés `plan`,
    `steps`, `summary`) pour rester compatible avec les consommateurs
    existants (ex. `router/chat_routes.py`). Les champs non pertinents
    (plan/steps) sont explicitement vides.
    """
    logger.info("[direct] Appel LLM unique (pas de planner/validator)")

    try:
        agent_output = await route_request("simple", user_input)
    except Exception as exc:
        logger.exception("[direct] Exception lors de l'appel LLM : %s", exc)
        return {
            "ok": False,
            "user_input": user_input,
            "intent_type": "direct",
            "response": "",
            "plan": [],
            "steps": [],
            "summary": {"total": 0, "succeeded": 0, "validated": 0, "failed": 1},
            "error": f"Exception : {exc!s}",
        }

    ok = bool(agent_output.get("ok")) if isinstance(agent_output, dict) else False
    response_text = _agent_text(agent_output) if ok else ""
    used = _agent_used(agent_output)

    if ok:
        logger.info("[direct] OK | provider=%s | %d chars",
                    used or "?", len(response_text))
    else:
        error_msg = (agent_output or {}).get("error", "Erreur inconnue") \
            if isinstance(agent_output, dict) else "Erreur inconnue"
        logger.warning("[direct] KO | %s", error_msg)

    result: Dict[str, Any] = {
        "ok": ok,
        "user_input": user_input,
        "intent_type": "direct",
        "response": response_text,
        "plan": [],
        "steps": [],
        "summary": {
            "total": 0,
            "succeeded": 1 if ok else 0,
            "validated": 0,
            "failed": 0 if ok else 1,
        },
    }
    if not ok:
        err = (agent_output or {}).get("error") if isinstance(agent_output, dict) else None
        result["error"] = err or "Appel LLM direct en échec."
    logger.info("=== Workflow END (direct) | ok=%s ===", ok)
    return result


def _dispatch_agent(task_type: str, description: str) -> Awaitable[Dict[str, Any]]:
    """Retourne la coroutine de l'agent approprié selon `task_type`."""
    if task_type == "code":
        return _run_code(description)
    if task_type == "analyse":
        return _run_analyse(description)
    # simple / moyen / complexe (ou inconnu → router gère le fallback)
    return _run_via_router(task_type, description)


# ---------------------------------------------------------------------------
# Helpers pour formater les sorties d'agent
# ---------------------------------------------------------------------------

def _agent_text(agent_output: Dict[str, Any]) -> str:
    """Extrait un texte lisible depuis la sortie d'agent (pour logs/validator)."""
    if not isinstance(agent_output, dict):
        return ""

    # code_agent / router → clé "text"
    text = agent_output.get("text")
    if isinstance(text, str) and text.strip():
        return text

    # analyse_agent → assemble summary + listes
    summary = agent_output.get("summary")
    if isinstance(summary, str) and summary.strip():
        parts = [summary.strip()]
        for key in ("observations", "patterns", "anomalies",
                    "insights", "recommendations"):
            items = agent_output.get(key) or []
            if items:
                parts.append(f"{key}: " + " | ".join(str(x) for x in items))
        return "\n".join(parts)

    return ""


def _agent_used(agent_output: Dict[str, Any]) -> str | None:
    """Provider réellement utilisé par l'agent, si disponible."""
    if isinstance(agent_output, dict):
        used = agent_output.get("used")
        if isinstance(used, str):
            return used
    return None


# ---------------------------------------------------------------------------
# Fonction publique
# ---------------------------------------------------------------------------

async def run_workflow(user_input: str) -> Dict[str, Any]:
    """Exécute le pipeline complet Planner → Router → Agent → Validator.

    Parameters
    ----------
    user_input : str
        La requête utilisateur à traiter.

    Returns
    -------
    dict
        {
            "ok": bool,
            "user_input": str,
            "plan": [...],     # sous-tâches renvoyées par le planner
            "steps": [         # un élément par sous-tâche
                {
                    "id": int,
                    "description": str,
                    "task_type": str,
                    "agent": {...},        # sortie brute de l'agent
                    "validation": {...},   # sortie du validator
                },
                ...
            ],
            "summary": {"total", "succeeded", "validated", "failed"},
            "error": str (optionnel, si échec global)
        }
    """
    logger.info("=== Workflow START ===")
    logger.info("Input : %s", (user_input or "").strip()[:200])

    # -----------------------------------------------------------------------
    # 0) Détection d'intention (mots-clés simples)
    # -----------------------------------------------------------------------
    intent_type = detect_intent(user_input or "")
    logger.info("Intent détecté : %s", intent_type)

    # Court-circuit "direct" : un seul appel LLM, pas de planner/validator.
    # La branche "analysis" ci-dessous reste strictement inchangée.
    if intent_type == "direct":
        return await _run_direct(user_input)

    # -----------------------------------------------------------------------
    # 1) PLANNER
    # -----------------------------------------------------------------------
    logger.info("[1/4] Planner : décomposition en sous-tâches…")
    plan_result = await plan_task(user_input)

    if not plan_result.get("ok"):
        error = plan_result.get("error", "Planner a échoué.")
        logger.error("Planner KO | %s", error)
        return {
            "ok": False,
            "user_input": user_input,
            "intent_type": intent_type,
            "plan": [],
            "steps": [],
            "summary": {"total": 0, "succeeded": 0, "validated": 0, "failed": 0},
            "error": f"Planner : {error}",
        }

    subtasks: List[Dict[str, Any]] = plan_result.get("subtasks", []) or []
    logger.info("Planner OK | %d sous-tâche(s)", len(subtasks))
    for st in subtasks:
        logger.info("  • [%s] (%s) %s",
                    st.get("id"), st.get("task_type"),
                    (st.get("description") or "")[:120])

    if not subtasks:
        return {
            "ok": False,
            "user_input": user_input,
            "intent_type": intent_type,
            "plan": [],
            "steps": [],
            "summary": {"total": 0, "succeeded": 0, "validated": 0, "failed": 0},
            "error": "Aucune sous-tâche générée.",
        }

    # -----------------------------------------------------------------------
    # 2-4) Pour chaque sous-tâche : Router → Agent → Validator
    # -----------------------------------------------------------------------
    steps: List[Dict[str, Any]] = []
    succeeded = 0
    validated_ok = 0
    failed = 0

    for st in subtasks:
        sid = st.get("id")
        task_type = st.get("task_type", "moyen")
        description = st.get("description", "")

        logger.info("--- Sous-tâche %s [%s] ---", sid, task_type)

        # ---- 2) ROUTER : choix de l'agent (via _dispatch_agent) -----------
        logger.info("[2/4] Router : dispatch vers agent '%s'", task_type)

        # ---- 3) AGENT -----------------------------------------------------
        logger.info("[3/4] Agent : exécution…")
        try:
            agent_output = await _dispatch_agent(task_type, description)
        except Exception as exc:
            logger.exception("Agent a levé une exception : %s", exc)
            agent_output = {
                "ok": False,
                "error": f"Exception : {exc!s}",
                "text": "",
            }

        agent_ok = bool(agent_output.get("ok"))
        used = _agent_used(agent_output)
        if agent_ok:
            succeeded += 1
            logger.info("Agent OK | provider=%s", used or "?")
        else:
            failed += 1
            logger.warning("Agent KO | %s",
                           agent_output.get("error", "Erreur inconnue"))

        # ---- 4) VALIDATOR -------------------------------------------------
        validation: Dict[str, Any]
        if agent_ok:
            logger.info("[4/4] Validator : évaluation…")
            text_to_validate = _agent_text(agent_output)
            if text_to_validate:
                try:
                    validation = await validate_response(text_to_validate)
                except Exception as exc:
                    logger.exception("Validator a levé une exception : %s", exc)
                    validation = {
                        "ok": False,
                        "valid": False,
                        "score": 0,
                        "error": f"Exception : {exc!s}",
                    }
            else:
                validation = {
                    "ok": False,
                    "valid": False,
                    "score": 0,
                    "error": "Sortie agent vide — rien à valider.",
                }

            if validation.get("ok") and validation.get("valid"):
                validated_ok += 1
                logger.info("Validator OK | valid=True score=%s used=%s",
                            validation.get("score"), validation.get("used"))
            else:
                logger.info("Validator | valid=%s score=%s reason=%s",
                            validation.get("valid"),
                            validation.get("score"),
                            (validation.get("reason") or
                             validation.get("error") or "")[:120])
        else:
            logger.info("[4/4] Validator : sauté (agent en échec)")
            validation = {
                "ok": False,
                "valid": False,
                "score": 0,
                "error": "Agent en échec — validation sautée.",
            }

        steps.append({
            "id": sid,
            "description": description,
            "task_type": task_type,
            "agent": agent_output,
            "validation": validation,
        })

    # -----------------------------------------------------------------------
    # Résumé
    # -----------------------------------------------------------------------
    summary = {
        "total": len(subtasks),
        "succeeded": succeeded,
        "validated": validated_ok,
        "failed": failed,
    }
    logger.info("=== Workflow END | %s ===", summary)

    return {
        "ok": succeeded > 0,
        "user_input": user_input,
        "intent_type": intent_type,
        "plan": subtasks,
        "steps": steps,
        "summary": summary,
    }
