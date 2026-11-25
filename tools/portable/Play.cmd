@echo off
if "%~1"=="" (
  echo Drag your game_scarlett_master.exe onto this file.
  pause
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Play.ps1" -GameExe "%~1" -PlaySession
if errorlevel 1 pause
