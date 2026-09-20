@echo off
REM modelclaw CLI launcher for Windows (Git Bash / CMD / PowerShell)
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0modelclaw.py" %*
) else (
    python "%~dp0modelclaw.py" %*
)
