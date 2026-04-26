"""Service de gestion de la configuration (clés API).

Stockage simple dans un fichier `.env` à la racine du projet.
Aucune sécurité avancée pour l'instant (clés en clair).
"""

from pathlib import Path
from typing import Dict, Optional


# Liste des clés supportées par l'application.
SUPPORTED_KEYS = [
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "CODESTRAL_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
]


class ConfigService:
    """Lit et écrit les clés API dans un fichier `.env`."""

    def __init__(self, env_path: Optional[Path] = None) -> None:
        # Racine du projet = parent du dossier `services/`.
        root = Path(__file__).resolve().parent.parent
        self.env_path: Path = env_path or (root / ".env")

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------
    def _read_all(self) -> Dict[str, str]:
        """Lit toutes les paires clé=valeur du fichier `.env`."""
        data: Dict[str, str] = {}
        if not self.env_path.exists():
            return data

        for raw_line in self.env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            data[key] = value
        return data

    def get_keys(self) -> Dict[str, str]:
        """Retourne uniquement les clés API supportées (chaîne vide si absent)."""
        data = self._read_all()
        return {k: data.get(k, "") for k in SUPPORTED_KEYS}

    # ------------------------------------------------------------------
    # Écriture
    # ------------------------------------------------------------------
    def save_keys(self, keys: Dict[str, Optional[str]]) -> Dict[str, str]:
        """Enregistre les clés fournies dans le fichier `.env`.

        - Conserve les autres variables d'environnement déjà présentes.
        - Une valeur `None` ou vide écrase la clé existante par une chaîne vide.
        - Ne garde que les clés supportées (filtre de sécurité basique).
        """
        current = self._read_all()

        for key in SUPPORTED_KEYS:
            if key in keys:
                value = keys[key]
                current[key] = value if value is not None else ""

        # Réécriture complète du fichier (format simple KEY=VALUE).
        lines = [f'{k}={v}' for k, v in current.items()]
        content = "\n".join(lines) + ("\n" if lines else "")
        self.env_path.write_text(content, encoding="utf-8")

        return self.get_keys()


# Instance partagée utilisée par les routes.
config_service = ConfigService()
