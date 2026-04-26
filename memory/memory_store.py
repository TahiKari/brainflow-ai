"""Stockage mémoire simple (en RAM).

Placeholder : dans une future version, ce module pourra être remplacé par
une base vectorielle, Redis, une base de données, etc.
"""

from typing import Any, Dict, Optional


class MemoryStore:
    """Store clé/valeur en mémoire pour les agents."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Enregistre une valeur."""
        self._store[key] = value

    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur (None si absente)."""
        return self._store.get(key)

    def delete(self, key: str) -> bool:
        """Supprime une valeur. Retourne True si la clé existait."""
        return self._store.pop(key, None) is not None

    def all(self) -> Dict[str, Any]:
        """Retourne une copie de tout le contenu."""
        return dict(self._store)

    def clear(self) -> None:
        """Vide la mémoire."""
        self._store.clear()
