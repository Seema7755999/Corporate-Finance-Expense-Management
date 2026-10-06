@echo off
setlocal
cd /d "%~dp0"
title LedgerLens
echo.
echo ==========================================================
echo   LedgerLens - one-click start (Windows)
echo ==========================================================
echo.

rem --- 1. Find Python: 3.11, 3.12 or 3.13 first, then any 3.10 or newer ---------
set "PY="
for %%V in (3.11 3.12 3.13) do (
    if not defined PY (
        where py >nul 2>nul && py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
    )
)
if not defined PY where py >nul 2>nul && py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && python -c "import sys" >nul 2>nul && set "PY=python"
if not defined PY goto :nopython
%PY% -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 goto :oldpython
%PY% -c "import sys; sys.exit(1 if sys.version_info[:2] >= (3, 14) else 0)" >nul 2>nul
if errorlevel 1 echo NOTE: this Python is newer than 3.13. LedgerLens was tested on 3.11, so installing may fail. Python 3.11 is the safe choice.

rem --- 2. Create the private Python environment (first time only) ---------------
if exist ".venv\Scripts\python.exe" goto :haveenv
echo [1/3] Creating a private Python environment (about 20 seconds)...
%PY% -m venv .venv
if errorlevel 1 goto :fail
:haveenv

rem --- 3. Install the packages (first time only, 5 to 20 minutes) ---------------
if exist ".venv\.packages_ok" goto :haveparts
echo [2/3] Installing the packages. The first time this takes 5 to 20 minutes. Please wait...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
echo done> ".venv\.packages_ok"
:haveparts

rem --- 4. Start the app ------------------------------------------------------------
echo [3/3] Starting LedgerLens. Your web browser opens by itself.
echo       Keep this window open while you use the app. Close it (or press Ctrl+C) to stop.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
goto :end

:nopython
echo Python was not found on this computer.
echo Install Python 3.11 from https://www.python.org/downloads/ and TICK the box
echo "Add python.exe to PATH" on the first installer screen. Then double-click this file again.
goto :end

:oldpython
echo Your Python is too old. Please install Python 3.11 from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH"), then double-click this file again.
goto :end

:fail
echo.
echo Something went wrong (see the message above). Open README.md and read the
echo Troubleshooting part, or copy the message above into a search engine.

:end
echo.
pause
