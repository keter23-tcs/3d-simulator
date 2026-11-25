@echo off
rem Start the 3D Simulator (menu navigated with numbers). On the first run everything is set up.
setlocal
cd /d "%~dp0"
title 3D Simulator
if exist ".venv\Scripts\python.exe" goto start
call "%~dp0Installieren.bat" --ohne-pause
if errorlevel 1 (
  pause
  exit /b 1
)
:start
".venv\Scripts\python.exe" "rechner\menue_3d.py"
endlocal
