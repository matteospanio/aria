@echo off
rem build.bat - native Windows build with MSVC (the Makefile's counterpart).
rem Run from an "x64 Native Tools Command Prompt for VS 2019/2022".
rem   build.bat        CPU build: build\aria.exe + build\libaria.lib
rem   build.bat test   CPU build + the hermetic unit tests, no model or GPU needed
rem   build.bat cuda   CUDA build: nvcc on PATH, ARIA_CUDA_ARCH (default sm_86)
setlocal enabledelayedexpansion
if not "%~1"=="" if /i not "%~1"=="test" if /i not "%~1"=="cuda" (
    echo usage: build.bat [test^|cuda]
    exit /b 1
)
cd /d "%~dp0"
if not exist build mkdir build

rem aria_win_compat.h is force-included: it supplies mmap, clock_gettime, __thread, ...
set "CFLAGS=/nologo /O2 /MD /Isrc /FIaria_win_compat.h"
set "CUDA_LINK="
if /i "%~1"=="cuda" (
    if not defined ARIA_CUDA_ARCH set "ARIA_CUDA_ARCH=sm_86"
    set "CFLAGS=!CFLAGS! /DARIA_CUDA"
    rem same nvcc flags as the Makefile: the DiT's CUDA-graph capture needs the per-thread stream
    nvcc -arch=!ARIA_CUDA_ARCH! -O3 --default-stream per-thread -Xcompiler /MD -Isrc -c src\aria_cuda.cu -o build\aria_cuda.obj || exit /b 1
    rem link with link.exe, not nvcc: nvcc's link step pulls in the static CRT under our /MD objects
    for %%i in (nvcc.exe) do set "CUDA_LINK=build\aria_cuda.obj /LIBPATH:"%%~dp$PATH:i..\lib\x64" cudart_static.lib cublas.lib"
)

rem libaria: every src\*.c except the CLI entry point and the POSIX-only HTTP server
set "SRCS="
set "OBJS="
for %%f in (src\*.c) do if /i not "%%~nxf"=="main.c" if /i not "%%~nxf"=="aria_server.c" (
    set "SRCS=!SRCS! %%f"
    set "OBJS=!OBJS! build\%%~nf.obj"
)
cl !CFLAGS! /MP /c /Fobuild\ !SRCS! || exit /b 1
lib /nologo /OUT:build\libaria.lib !OBJS! || exit /b 1

cl !CFLAGS! /c /Fobuild\ src\main.c || exit /b 1
link /nologo /OUT:build\aria.exe build\main.obj build\libaria.lib !CUDA_LINK! || exit /b 1

if /i "%~1"=="test" (
    rem the Makefile's TESTS list, so the two can't drift
    set "TESTS="
    for /f "tokens=2 delims==" %%l in ('findstr /b /c:"TESTS :=" Makefile') do set "TESTS=%%l"
    if not defined TESTS (echo no TESTS list found in the Makefile & exit /b 1)
    for %%t in (!TESTS!) do (
        echo ==^> build %%t
        cl !CFLAGS! /Fobuild\ /Febuild\%%t.exe tests\%%t.c build\libaria.lib || exit /b 1
        echo ==^> run %%t
        build\%%t.exe || exit /b 1
    )
    echo all tests passed
)
