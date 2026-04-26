"""Point d'entrée de l'application BrainFlow AI.

Lance un serveur FastAPI minimal et modulaire.
Aucun code IA n'est présent : ce fichier assemble uniquement les routes.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from router.api_router import api_router
from router.chat import router as chat_public_router

# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BrainFlow AI",
    description="Application multi-agent IA (squelette modulaire, sans IA).",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Routes racines
# ---------------------------------------------------------------------------
@app.get("/", tags=["root"])
def root():
    """Message de bienvenue."""
    return {
        "app": "BrainFlow AI",
        "version": "0.1.0",
        "message": "Bienvenue. Voir /docs pour l'API et /ui pour l'interface.",
    }


@app.get("/health", tags=["root"])
def health():
    """Contrôle de santé du service."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# API (préfixe /api)
# ---------------------------------------------------------------------------
app.include_router(api_router)

# Endpoint public `/chat` (racine) — réponse multi-agent
app.include_router(chat_public_router)


# ---------------------------------------------------------------------------
# Interface utilisateur (statique)
# ---------------------------------------------------------------------------
UI_DIR = Path(__file__).parent / "ui"

if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


@app.get("/ui-home", include_in_schema=False)
def ui_home():
    """Redirige vers la page d'accueil HTML (si besoin d'un accès direct)."""
    index = UI_DIR / "index.html"
    return FileResponse(str(index))


# ---------------------------------------------------------------------------
# Entrée locale (optionnelle)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
