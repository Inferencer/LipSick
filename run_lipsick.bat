@echo off
REM Change to the directory of the batch file
cd /d "%~dp0"

REM Activate the LipSick environment
call conda activate LipSick

REM Run app.py
python app.py

