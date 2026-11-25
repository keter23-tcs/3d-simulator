@echo off
rem Set up the 3D Simulator: everything in one go (Python if needed, packages, Input and Output folders).
rem A double click is enough. Start.bat calls this script by itself on the first run.
setlocal
cd /d "%~dp0"
title 3D Simulator setup
echo.
echo  ============================================================
echo    3D Simulator - Setup
echo  ============================================================
echo.

rem --- create folders ---
if not exist "Input" mkdir "Input"
if not exist "Output" mkdir "Output"

rem --- look for Python 3.10 or newer ---
set PY=
call :suche
if defined PY goto python_ok

echo  Python was not found. It will now be installed automatically ^(winget, for your user only, about 1 minute^) ...
where winget >nul 2>nul
if errorlevel 1 goto ohne_winget
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
call :suche
if defined PY goto python_ok

:ohne_winget
echo.
echo  Python could not be installed automatically.
echo  Please install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH" at the bottom^)
echo  and then double-click Installieren.bat again.
start "" "https://www.python.org/downloads/"
goto fehler

:python_ok
echo  Python: %PY%

rem --- private environment (.venv) and packages ---
if not exist ".venv\Scripts\python.exe" (
  echo  Creating the Python environment ...
  "%PY%" -m venv .venv
  if errorlevel 1 goto fehler
)
echo  Installing the packages ^(numpy, scipy, scikit-fem, pyamg; about 100 MB, needs internet^) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fehler

rem --- self test ---
".venv\Scripts\python.exe" -c "import numpy, scipy, skfem, pyamg; print('  Self test: all packages present')"
if errorlevel 1 goto fehler

rem --- optional: NVIDIA GPU ---
where nvidia-smi >nul 2>nul
if errorlevel 1 goto fertig
".venv\Scripts\python.exe" -c "import cupy" >nul 2>nul
if not errorlevel 1 goto fertig
echo.
echo  An NVIDIA graphics card was found. With GPU acceleration the simulator calculates faster
echo  ^(download about 1-2 GB^). You can also do this later in the menu under [8].
choice /c YN /t 20 /d N /m "  Install GPU support now"
if errorlevel 255 goto fertig
if errorlevel 2 goto fertig
".venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt
if errorlevel 1 echo  The GPU installation failed, the simulator will calculate on the CPU instead.

:fertig
echo.
echo  ============================================================
echo    Done! The 3D Simulator is set up.
echo    Input folder: the jobs from the app go here
echo    Output folder: the results end up here
echo    To start: double-click Start.bat
echo  ============================================================
if /i not "%~1"=="--ohne-pause" if /i not "%~1"=="--no-pause" pause
exit /b 0

:fehler
echo.
echo  The setup failed ^(see message above^). Check your internet connection and try again.
if exist ".venv" if not exist ".venv\Scripts\pip.exe" rmdir /s /q ".venv"
if /i not "%~1"=="--ohne-pause" if /i not "%~1"=="--no-pause" pause
exit /b 1

:suche
rem Look for Python 3.10+: the py launcher, python on PATH, then the usual install location of winget/python.org
for %%C in ("py -3" "python") do (
  if not defined PY (
    %%~C -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if not errorlevel 1 for /f "delims=" %%P in ('%%~C -c "import sys; print(sys.executable)"') do set "PY=%%P"
  )
)
if defined PY exit /b 0
for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do (
  if exist "%%D\python.exe" if not defined PY (
    "%%D\python.exe" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=%%D\python.exe"
  )
)
exit /b 0
