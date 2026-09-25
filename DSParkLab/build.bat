@echo off
setlocal
cd /d "%~dp0"

rem Builds DSParkLab.exe with MSVC. Run from any prompt: a Developer Command
rem Prompt is used as-is; otherwise vswhere finds the newest Visual Studio
rem (2019 or later, any edition, or the Build Tools) with the C++ x64 tools.

where cl >nul 2>&1
if %ERRORLEVEL% == 0 goto build

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo Visual Studio was not found. Install Visual Studio 2019 or later
    echo ^(or the Build Tools^) with the "Desktop development with C++" workload.
    exit /b 1
)
set "VSDIR="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
    echo No Visual Studio installation has the C++ x64 build tools.
    echo Add the "Desktop development with C++" workload in the Visual Studio Installer.
    exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
if errorlevel 1 (
    echo Could not set up the MSVC x64 environment from "%VSDIR%".
    exit /b 1
)

:build
echo Building DSParkLab...

cl /std:c++20 /O2 /W4 /WX- /EHsc /MP /DUNICODE /DNOMINMAX ^
    /I.. /Ivendor /Ivendor/imgui ^
    main.cpp ^
    vendor/imgui/imgui.cpp ^
    vendor/imgui/imgui_draw.cpp ^
    vendor/imgui/imgui_tables.cpp ^
    vendor/imgui/imgui_widgets.cpp ^
    vendor/imgui/imgui_impl_win32.cpp ^
    vendor/imgui/imgui_impl_dx11.cpp ^
    /Fe:DSParkLab.exe /nologo ^
    /link d3d11.lib d3dcompiler.lib user32.lib gdi32.lib ole32.lib comdlg32.lib

if %ERRORLEVEL% == 0 (
    echo Build OK: DSParkLab.exe
    del *.obj >nul 2>&1
) else (
    echo BUILD FAILED
    exit /b 1
)
