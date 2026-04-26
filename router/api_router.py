"""Router principal de l'API.

Agrège tous les sous-routers (agents, etc.) sous le préfixe `/api`.
"""

from fastapi import APIRouter

from router.agents_routes import router as agents_router
from router.chat_routes import router as chat_router
from router.config_routes import router as config_router

api_router = APIRouter(prefix="/api")

# Enregistrement des sous-routers
api_router.include_router(agents_router)
api_router.include_router(chat_router)
api_router.include_router(config_router)
