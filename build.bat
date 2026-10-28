@echo off
rem ============================================================================
rem  Builds gpumemtest.exe. Requires CUDA Toolkit (nvcc) and Visual Studio C++ (cl.exe).
rem  Default target: sm_89 (RTX 40xx incl. 4090) + PTX for newer GPUs.
rem  Other GPU, e.g. RTX 30xx:   set ARCH=-arch=sm_86
rem                              build.bat
rem  (this file must stay ASCII-only: cmd.exe misparses UTF-8 batch files)
rem ============================================================================
setlocal
cd /d "%~dp0"

if not defined ARCH set "ARCH=-gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89"

set "NVCC=nvcc"
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "NVCC=%CUDA_PATH%\bin\nvcc.exe"

rem cl.exe is not in PATH: find Visual Studio with vswhere and load its environment
where cl >nul 2>nul && goto :build
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :no_vs
for /f "usebackq delims=" %%i in (`call "%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR goto :no_vs
rem vcvars64.bat itself calls vswhere.exe by name
set "PATH=%PATH%;%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
where cl >nul 2>nul || goto :no_vs

:build
echo [build] "%NVCC%" %ARCH%
"%NVCC%" -O3 -std=c++17 %ARCH% -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -Xcompiler "/EHsc /W3 /nologo" -o gpumemtest.exe gpumemtest.cu
if errorlevel 1 goto :failed
echo [build] OK: %~dp0gpumemtest.exe
exit /b 0

:failed
echo [build] FAILED
exit /b 1

:no_vs
echo [build] Visual Studio C++ tools (cl.exe) not found.
echo         Install the "Desktop development with C++" workload or run from "x64 Native Tools Command Prompt".
exit /b 1
