"""Route chat : expose `run_workflow` via POST /api/chat.

Simple et fonctionnel : l'UI envoie `{message: "..."}`, le serveur exécute
le pipeline complet (Planner → Router → Agent → Validator) et renvoie
une réponse texte lisible + le détail structuré.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.workflow import run_workflow


router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Schémas Pydantic
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., description="Message utilisateur à traiter.")


# ---------------------------------------------------------------------------
# Helpers — extraction d'un texte d'affichage depuis la sortie du workflow
# ---------------------------------------------------------------------------

def _step_text(step: Dict[str, Any]) -> str:
    """Extrait un texte lisible d'une sous-tâche du workflow."""
    agent = step.get("agent") or {}
    if not isinstance(agent, dict):
        return ""

    # Cas 1 : agent renvoie directement "text"
    text = agent.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Cas 2 : analyse_agent → summary + éventuelles listes
    summary = agent.get("summary")
    if isinstance(summary, str) and summary.strip():
        parts: List[str] = [summary.strip()]
        for key in ("observations", "patterns", "anomalies",
                    "insights", "recommendations"):
            items = agent.get(key) or []
            if items:
                parts.append(
                    f"**{key.capitalize()}** : "
                    + " | ".join(str(x) for x in items)
                )
        return "\n\n".join(parts)

    # Cas 3 : erreur agent
    err = agent.get("error")
    if isinstance(err, str) and err.strip():
        return f"⚠️ {err.strip()}"

    return ""


def _build_response_text(workflow_result: Dict[str, Any]) -> str:
    """Construit un texte de réponse unique à partir des steps du workflow."""
    steps = workflow_result.get("steps") or []
    if not steps:
        return ""

    # Une seule sous-tâche → on renvoie son texte tel quel.
    if len(steps) == 1:
        return _step_text(steps[0])

    # Plusieurs sous-tâches → on concatène avec un titre par étape.
    blocks: List[str] = []
    for st in steps:
        sid = st.get("id", "?")
        desc = (st.get("description") or "").strip()
        ttype = st.get("task_type", "?")
        text = _step_text(st) or "(pas de sortie)"
        header = f"### Étape {sid} — {desc}  _(type : {ttype})_"
        blocks.append(f"{header}\n\n{text}")
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("")
async def chat(payload: ChatRequest) -> Dict[str, Any]:
    """Exécute `run_workflow` sur le message utilisateur.

    Retourne :
        {
            "ok": bool,
            "message": str,       # message d'entrée
            "response": str,      # texte lisible prêt à afficher
            "plan": [...],        # sous-tâches du planner
            "steps": [...],       # détail par sous-tâche (agent + validator)
            "summary": {...},     # stats (total / succeeded / validated / failed)
            "error": str          # présent uniquement en cas d'échec
        }
    """
    user_message = (payload.message or "").strip()
    if not user_message:
        return {
            "ok": False,
            "message": "",
            "response": "",
            "plan": [],
            "steps": [],
            "summary": {"total": 0, "succeeded": 0, "validated": 0, "failed": 0},
            "error": "Message vide.",
        }

    result = await run_workflow(user_message)

    # Mode "direct" : le workflow renvoie directement le texte final
    # dans `response` (pas de steps, pas d'analyse).
    # Pour les autres modes on reconstruit le texte depuis les steps.
    direct_text = result.get("response") if isinstance(result, dict) else None
    if isinstance(direct_text, str) and direct_text.strip():
        response_text = direct_text
    else:
        response_text = _build_response_text(result)
        if not response_text:
            response_text = result.get("error") or "Aucune réponse générée."

    return {
        "ok": bool(result.get("ok")),
        "message": user_message,
        "response": response_text,
        "plan": result.get("plan", []),
        "steps": result.get("steps", []),
        "summary": result.get("summary", {}),
        **({"error": result["error"]} if result.get("error") else {}),
    }
