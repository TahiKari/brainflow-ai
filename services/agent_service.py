"""Service d'orchestration des agents.

Placeholder : ce service expose une API interne utilisée par les routes.
Il ne contient aucune logique IA pour l'instant.
"""

from typing import Any, Dict, List

from memory.memory_store import MemoryStore


class AgentService:
    """Orchestre l'enregistrement et l'exécution des agents."""

    def __init__(self) -> None:
        self._agents: Dict[str, Dict[str, Any]] = {}
        self.memory = MemoryStore()

    def register_agent(self, name: str, description: str = "") -> Dict[str, Any]:
        """Enregistre un agent (placeholder)."""
        agent = {"name": name, "description": description}
        self._agents[name] = agent
        return agent

    def list_agents(self) -> List[Dict[str, Any]]:
        """Retourne la liste des agents enregistrés."""
        return list(self._agents.values())

    def get_agent(self, name: str) -> Dict[str, Any] | None:
        """Retourne un agent par son nom."""
        return self._agents.get(name)

    def run_agent(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute un agent (placeholder : aucune IA).

        Retourne un écho du payload avec des métadonnées.
        """
        agent = self.get_agent(name)
        if agent is None:
            return {
                "status": "error",
                "message": f"Agent '{name}' introuvable.",
            }

        return {
            "status": "ok",
            "agent": name,
            "input": payload,
            "output": None,
            "message": "Exécution simulée (aucune IA branchée).",
        }


# Instance partagée (singleton simple) utilisée par les routes.
agent_service = AgentService()
