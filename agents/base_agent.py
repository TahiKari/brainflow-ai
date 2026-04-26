"""Classe de base pour tous les agents IA.

Ce fichier est un placeholder : aucune logique IA n'est implémentée ici.
Il définit uniquement l'interface que chaque agent devra respecter.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Interface commune à tous les agents."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute l'agent avec un payload donné.

        À implémenter dans les sous-classes.
        """
        raise NotImplementedError

    def info(self) -> Dict[str, str]:
        """Retourne des métadonnées sur l'agent."""
        return {
            "name": self.name,
            "description": self.description,
        }
