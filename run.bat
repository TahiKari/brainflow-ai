@echo off
REM Script de lancement pour Windows (cmd.exe / PowerShell)
REM Usage : run.bat

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [!] Environnement virtuel introuvable. Creation...
    python -m venv venv
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\pip.exe install -r requirements.txt
)

echo [+] Lancement du serveur BrainFlow AI sur http://127.0.0.1:8000
venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
