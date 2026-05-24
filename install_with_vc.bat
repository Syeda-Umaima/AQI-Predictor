@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
  echo Failed to initialize MSVC environment
  exit /b 1
)
"%CD%\.venv\Scripts\python.exe" -m pip install -r requirements.txt
endlocal
