"""Routes de configuration (clés API).

Utilise le gestionnaire `services.api_key_manager` (stockage JSON).
"""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from services.api_key_manager import load_api_keys, save_api_keys


router = APIRouter(prefix="/config", tags=["config"])

# Liste des clés attendues par l'UI (affichage systématique même si absentes).
SUPPORTED_KEYS = [
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "CODESTRAL_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
]


class ApiKeysUpdate(BaseModel):
    """Modèle de mise à jour des clés API. Toutes les clés sont optionnelles."""

    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None
    CODESTRAL_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None


def _normalize(stored: dict) -> dict:
    """Retourne les clés attendues (chaîne vide si absente)."""
    return {k: (stored.get(k) or "") for k in SUPPORTED_KEYS}


@router.get("/keys")
def get_keys():
    """Retourne les clés API enregistrées (vides si non définies)."""
    stored = load_api_keys()
    return {"status": "ok", "keys": _normalize(stored)}


@router.post("/keys")
def save_keys(data: ApiKeysUpdate):
    """Enregistre les clés API fournies via `save_api_keys` (api_keys.json)."""
    # On ne transmet que les champs fournis explicitement, et on normalise
    # les `None` en chaîne vide pour être cohérent côté lecture.
    payload = {
        k: (v if v is not None else "")
        for k, v in data.model_dump(exclude_unset=True).items()
    }
    saved = save_api_keys(payload)
    return {
        "status": "ok",
        "message": "Clés enregistrées avec succès.",
        "keys": _normalize(saved),
    }
