@echo off
REM One-time installation into the current Python (use a virtual environment if you prefer).
cd /d "%~dp0"
python -m pip install --upgrade pip
python -m pip install -e ".[app]"
python -m mf62pinn --version
pause
