@echo off
title KuroLine AI Anime Studio Launcher
cd /d "%~dp0"

echo ============================================================
echo   KUROLINE AI STUDIO - RTX 4070 Ti ACCELERATED RUNTIME
echo ============================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH!
    pause
    exit /b 1
)

:: Free port 8000 if an old process is hanging onto it
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :8000 ^| findstr LISTENING') do (
    echo [*] Clearing previous process on port 8000 (PID: %%a)...
    taskkill /F /PID %%a >nul 2>&1
)

:: Background watchdog that waits for server readiness and forces browser to open
start "" /min powershell -NoProfile -ExecutionPolicy Bypass -Command "$u='http://127.0.0.1:8000'; for ($i=0; $i -lt 50; $i++) { Start-Sleep -Milliseconds 300; try { $req = [System.Net.WebRequest]::Create($u + '/api/status'); $resp = $req.GetResponse(); if ($resp.StatusCode -eq 200) { $resp.Close(); break; } } catch {} }; Start-Process $u"

echo [*] Starting KuroLine AI Server on http://127.0.0.1:8000 ...
python server.py
pause
