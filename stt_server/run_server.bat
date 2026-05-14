@echo off
REM Launcher for the Jarvis STT GPU offload server.
REM Used by the Task Scheduler autostart on MEDIA-HOST (M36).
REM Portable — %~dp0 resolves to this file's directory at runtime,
REM so the bat works from any clone location.
cd /d "%~dp0"
venv\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1
