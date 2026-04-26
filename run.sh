#!/usr/bin/env bash
# Script de lancement pour Git Bash / MINGW64 / Linux / macOS
# Usage : ./run.sh

set -e

cd "$(dirname "$0")"

# Détection de l'emplacement de python dans le venv (Windows vs Unix)
if [ -f "venv/Scripts/python.exe" ]; then
    PY="venv/Scripts/python.exe"
elif [ -f "venv/bin/python" ]; then
    PY="venv/bin/python"
else
    echo "[!] Environnement virtuel introuvable. Création..."
    python -m venv venv
    if [ -f "venv/Scripts/python.exe" ]; then
        PY="venv/Scripts/python.exe"
    else
        PY="venv/bin/python"
    fi
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements.txt
fi

echo "[+] Lancement du serveur BrainFlow AI sur http://127.0.0.1:8000"
"$PY" -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
