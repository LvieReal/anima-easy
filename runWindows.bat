@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 (
    echo The project folder could not be opened.
    pause
    exit /b 1
)
set "PYTHONUTF8=1"
set "uvCommand=uv"

where uv >nul 2>nul
if not errorlevel 1 goto launch
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "uvCommand=%USERPROFILE%\.local\bin\uv.exe"
    goto launch
)
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe" (
    set "uvCommand=%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe"
    goto launch
)
echo uv was not found. Install it once, then run this launcher again.
echo https://docs.astral.sh/uv/getting-started/installation/
popd
pause
exit /b 127

:launch
"%uvCommand%" run --locked --no-dev anima-easy %*
set "result=%errorlevel%"
popd
if "%result%"=="0" exit /b 0
echo.
echo Anima Easy could not start or save the session. See the message above.
pause
exit /b %result%
