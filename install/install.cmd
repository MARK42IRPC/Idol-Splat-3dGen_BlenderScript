@echo off
rem Double-click entry point: forwards every argument to install.ps1.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if errorlevel 1 (
  echo.
  echo IS3D install failed. See the messages above.
  pause
)
endlocal
