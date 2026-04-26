"""Endpoint public `/chat` — réponse multi-agent.

Point d'entrée unique et minimal pour l'utilisateur :
- Input  : { "message": "..." }
- Sortie : réponse multi-agent (texte final + détail par agent)

Cet endpoint est volontairement **séparé** de `router/chat_routes.py`
(qui lui reste exposé sous `/api/chat`) afin de ne modifier aucun
code existant. Il réutilise `services.workflow.run_workflow`, qui
bénéficie déjà :
- du cache LLM (`memory.cache`)  → coût réduit sur requêtes répétées,
- des providers gratuits/low-cost (Groq, Gemini, Mistral free, OpenRouter).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from services.workflow import run_workflow


# ---------------------------------------------------------------------------
# Router : exposé à la racine (pas de préfixe `/api`)
# ---------------------------------------------------------------------------

router = APIRouter(tags=["chat"])


# ---------------------------------------------------------------------------
# Schémas (I/O)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """Corps de la requête : un unique champ `message`."""
    message: str = Field(..., description="Message utilisateur à traiter.")


class AgentStep(BaseModel):
    """Résultat d'un agent sur une sous-tâche (version allégée)."""
    id: int
    description: str
    task_type: str
    provider: Optional[str] = None
    text: str = ""
    ok: bool = False
    valid: Optional[bool] = None
    score: Optional[int] = None


class ChatResponse(BaseModel):
    """Réponse multi-agent renvoyée au client."""
    ok: bool
    message: str
    response: str                     # texte final consolidé
    agents: List[AgentStep]           # détail par agent
    summary: Dict[str, int]           # {total, succeeded, validated, failed}


# ---------------------------------------------------------------------------
# Helpers (purs, testables unitairement)
# ---------------------------------------------------------------------------

def _step_text(agent_output: Dict[str, Any]) -> str:
    """Extrait un texte lisible depuis la sortie brute d'un agent."""
    if not isinstance(agent_output, dict):
        return ""

    # Cas standard : clé "text"
    text = agent_output.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    # analyse_agent : "summary" + listes
    summary = agent_output.get("summary")
    if isinstance(summary, str) and summary.strip():
        parts: List[str] = [summary.strip()]
        for key in ("observations", "patterns", "anomalies",
                    "insights", "recommendations"):
            items = agent_output.get(key) or []
            if items:
                parts.append(
                    f"{key.capitalize()} : "
                    + " | ".join(str(x) for x in items)
                )
        return "\n\n".join(parts)

    # Erreur explicite
    err = agent_output.get("error")
    if isinstance(err, str) and err.strip():
        return f"⚠️ {err.strip()}"

    return ""


def _build_final_response(steps: List[Dict[str, Any]]) -> str:
    """Assemble le texte final à partir des sous-tâches."""
    if not steps:
        return ""

    if len(steps) == 1:
        return _step_text(steps[0].get("agent") or {})

    blocks: List[str] = []
    for st in steps:
        desc = (st.get("description") or "").strip()
        text = _step_text(st.get("agent") or {}) or "(pas de sortie)"
        blocks.append(f"### {desc}\n\n{text}")
    return "\n\n---\n\n".join(blocks)


def _to_agent_steps(steps: List[Dict[str, Any]]) -> List[AgentStep]:
    """Convertit les steps internes du workflow en `AgentStep` publics."""
    out: List[AgentStep] = []
    for st in steps:
        agent = st.get("agent") or {}
        validation = st.get("validation") or {}
        out.append(AgentStep(
            id=int(st.get("id") or 0),
            description=str(st.get("description") or ""),
            task_type=str(st.get("task_type") or ""),
            provider=agent.get("used") if isinstance(agent.get("used"), str) else None,
            text=_step_text(agent),
            ok=bool(agent.get("ok")),
            valid=validation.get("valid") if isinstance(validation, dict) else None,
            score=validation.get("score") if isinstance(validation, dict) else None,
        ))
    return out


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatMessage) -> ChatResponse:
    """Exécute le pipeline multi-agent sur le message utilisateur.

    Renvoie un texte final consolidé **et** le détail par agent pour
    permettre à un client (UI ou tests) d'inspecter la qualité de chaque
    étape.
    """
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le champ 'message' ne peut pas être vide.",
        )

    result = await run_workflow(message)

    steps = result.get("steps") or []
    final_text = _build_final_response(steps) or (
        result.get("error") or "Aucune réponse générée."
    )

    return ChatResponse(
        ok=bool(result.get("ok")),
        message=message,
        response=final_text,
        agents=_to_agent_steps(steps),
        summary=result.get("summary") or {
            "total": 0, "succeeded": 0, "validated": 0, "failed": 0,
        },
    )
