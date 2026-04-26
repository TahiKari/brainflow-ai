"""Gestionnaire simple et fiable des clés API.

Stockage : fichier JSON `api_keys.json` à la racine du projet.

API publique :
    - load_api_keys() -> dict
    - save_api_keys(data: dict) -> dict

Fiabilité :
    - Écriture atomique (écriture dans un fichier temporaire puis remplacement).
    - Tolérance aux fichiers manquants / JSON corrompu (retourne {}).
    - Encodage UTF-8 systématique.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Emplacement du fichier de stockage
# ---------------------------------------------------------------------------
# Racine du projet = parent du dossier `services/`.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
API_KEYS_PATH: Path = _PROJECT_ROOT / "api_keys.json"


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------
def load_api_keys() -> Dict[str, Any]:
    """Charge les clés API depuis `api_keys.json`.

    Retourne un dictionnaire vide si le fichier n'existe pas ou est invalide.
    Cette fonction ne lève jamais d'exception liée au stockage.
    """
    if not API_KEYS_PATH.exists():
        return {}

    try:
        with API_KEYS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Fichier corrompu ou illisible → on repart sur un état vide.
        return {}

    if not isinstance(data, dict):
        # Format inattendu (ex : liste) → ignoré.
        return {}

    return data


# ---------------------------------------------------------------------------
# Écriture
# ---------------------------------------------------------------------------
def save_api_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Enregistre / met à jour les clés API dans `api_keys.json`.

    - Fusionne `data` avec les clés déjà présentes (les nouvelles valeurs
      écrasent les anciennes ; les clés non fournies sont conservées).
    - Écriture atomique : on écrit dans un fichier temporaire dans le même
      répertoire, puis on fait un `os.replace` pour basculer, évitant
      ainsi tout fichier tronqué en cas de crash.

    Retourne le dictionnaire final enregistré.
    """
    if not isinstance(data, dict):
        raise TypeError("save_api_keys(data) attend un dictionnaire.")

    merged: Dict[str, Any] = load_api_keys()
    merged.update(data)

    # S'assurer que le répertoire parent existe.
    API_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Écriture atomique via fichier temporaire dans le même répertoire.
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".api_keys_", suffix=".tmp", dir=str(API_KEYS_PATH.parent)
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, API_KEYS_PATH)
    except Exception:
        # Nettoyage du fichier temporaire en cas d'échec.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    return merged
