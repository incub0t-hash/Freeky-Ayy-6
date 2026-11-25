@echo off
start /wait "" "%~dp0prerequisites\vc_redist.x64.exe" /install /passive /norestart
exit /b %errorlevel%
