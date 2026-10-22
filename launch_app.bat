@echo off
REM Opens the MF6.2 PINN graphical tool in your browser.
REM First time only:  python -m pip install -e ".[app]"
cd /d "%~dp0"
python -m mf62pinn app
if errorlevel 1 (
  echo.
  echo If this failed, install the package first:   python -m pip install -e ".[app]"
  pause
)
