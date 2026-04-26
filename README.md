# BrainFlow AI

Application multi-agent IA construite avec **FastAPI**.

Ce dépôt contient la **structure de base** du projet : aucun code IA n'est encore implémenté, seulement les modules modulaires prêts à accueillir la logique.

## Structure

```
BrainFlow AI/
├── main.py                 # Point d'entrée FastAPI
├── requirements.txt        # Dépendances Python
├── run.bat                 # Script de lancement Windows (cmd.exe)
├── run.sh                  # Script de lancement Bash / Git Bash / Linux / macOS
├── agents/                 # Définition des agents IA
│   └── base_agent.py
├── router/                 # Routes de l'API
│   ├── api_router.py
│   └── agents_routes.py
├── services/               # Logique métier / orchestration
│   └── agent_service.py
├── memory/                 # Couche mémoire (stockage contextuel)
│   └── memory_store.py
└── ui/                     # Interface utilisateur
    └── index.html
```

## 🚀 Lancement rapide (recommandé)

Les scripts `run.bat` et `run.sh` créent automatiquement l'environnement virtuel (si absent), installent les dépendances et démarrent le serveur.

### Sur Windows (cmd.exe / PowerShell / double-clic)

```cmd
run.bat
```

### Sur Git Bash / MINGW64 / Linux / macOS

```bash
./run.sh
```

> 💡 Si `./run.sh` refuse de s'exécuter : `chmod +x run.sh` puis relancer.

## 🛠️ Installation manuelle

### 1. Créer l'environnement virtuel

```bash
python -m venv venv
```

### 2. Activer l'environnement

| Shell                           | Commande                         |
|---------------------------------|----------------------------------|
| **cmd.exe** (Windows)           | `venv\Scripts\activate.bat`      |
| **PowerShell** (Windows)        | `venv\Scripts\Activate.ps1`      |
| **Git Bash / MINGW64**          | `source venv/Scripts/activate`   |
| **Linux / macOS**               | `source venv/bin/activate`       |

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer le serveur

⚠️ **Important** : le module est `main:app` (pas `app:main` ni `app:app`).

```bash
uvicorn main:app --reload
```

Ou sans activer le venv, selon le shell :

| Shell                   | Commande                                                        |
|-------------------------|-----------------------------------------------------------------|
| **cmd.exe**             | `venv\Scripts\python.exe -m uvicorn main:app --reload`          |
| **Git Bash / MINGW64**  | `venv/Scripts/python.exe -m uvicorn main:app --reload`          |
| **Linux / macOS**       | `venv/bin/python -m uvicorn main:app --reload`                  |

> ❗ **Attention (Git Bash)** : n'utilisez **pas** les backslashes `\` dans les chemins car Bash les interprète comme caractères d'échappement. Utilisez toujours des slashes `/`.

## 🌐 Accès au serveur

Une fois démarré, le serveur écoute sur http://127.0.0.1:8000

- 🏠 Accueil JSON : http://127.0.0.1:8000/
- 💚 Santé : http://127.0.0.1:8000/health
- 🎨 Interface UI : http://127.0.0.1:8000/ui/
- 📖 Docs interactives (Swagger) : http://127.0.0.1:8000/docs
- 📘 Redoc : http://127.0.0.1:8000/redoc

## 📡 Endpoints disponibles

| Méthode | URL                     | Description                          |
|---------|-------------------------|--------------------------------------|
| GET     | `/`                     | Message de bienvenue                 |
| GET     | `/health`               | Statut du service                    |
| GET     | `/api/agents`           | Liste des agents                     |
| POST    | `/api/agents`           | Enregistre un agent                  |
| GET     | `/api/agents/{name}`    | Récupère un agent par nom            |
| POST    | `/api/agents/run`       | Lance un agent (placeholder)         |
| GET     | `/ui/`                  | Interface web                        |

## 🧩 Étapes suivantes (à venir)

- Brancher une vraie logique IA dans `agents/`
- Remplacer `memory_store.py` par une base vectorielle
- Ajouter des tests unitaires dans `/tests`

## Licence

MIT
