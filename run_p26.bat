@echo off
REM Full P26 identification from the command line. Results in runs\p26\report.html
cd /d "%~dp0"
python -m mf62pinn fit -c configs\p26.yaml
if exist runs\p26\report.html start "" runs\p26\report.html
pause
