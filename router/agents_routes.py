"""Routes relatives aux agents."""

from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from services.agent_service import agent_service

router = APIRouter(prefix="/agents", tags=["agents"])


class AgentCreate(BaseModel):
    name: str
    description: str = ""


class AgentRunRequest(BaseModel):
    name: str
    payload: Dict[str, Any] = {}


@router.get("")
def list_agents():
    """Retourne la liste des agents enregistrés."""
    return {"agents": agent_service.list_agents()}


@router.post("", status_code=201)
def create_agent(data: AgentCreate):
    """Enregistre un nouvel agent (placeholder)."""
    agent = agent_service.register_agent(data.name, data.description)
    return {"status": "ok", "agent": agent}


@router.get("/{name}")
def get_agent(name: str):
    """Retourne un agent par son nom."""
    agent = agent_service.get_agent(name)
    if agent is None:
        return {"status": "error", "message": f"Agent '{name}' introuvable."}
    return {"status": "ok", "agent": agent}


@router.post("/run")
def run_agent(data: AgentRunRequest):
    """Exécute un agent (aucune logique IA pour l'instant)."""
    return agent_service.run_agent(data.name, data.payload)
