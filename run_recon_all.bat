@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  run_recon_all.bat
::  Runs FreeSurfer recon-all on a T1w MRI scan via Docker.
::
::  Usage:
::    run_recon_all.bat <input_file.nii.gz> <subject_id>
::
::  Example:
::    run_recon_all.bat sub-116_T1w.nii.gz sub-116
::
::  Prerequisites:
::    - Docker Desktop running (WSL2 backend recommended)
::    - FreeSurfer license.txt placed in .\license\license.txt
::    - Input .nii.gz placed in .\input\
:: ============================================================

:: ============================================================
::  CONFIGURATION — adjust these to match your machine
:: ============================================================

:: Number of CPU cores to give recon-all.
:: Rule of thumb: (your total cores - 2) to keep the system responsive.
:: Check your core count: Task Manager -> Performance -> CPU -> Cores
set NUM_CORES=12

:: ============================================================

:: --- Argument validation ------------------------------------
if "%~1"=="" (
    echo [ERROR] No input file specified.
    echo Usage: run_recon_all.bat ^<filename.nii.gz^> ^<subject_id^>
    exit /b 1
)
if "%~2"=="" (
    echo [ERROR] No subject ID specified.
    echo Usage: run_recon_all.bat ^<filename.nii.gz^> ^<subject_id^>
    exit /b 1
)

set INPUT_FILE=%~1
set SUBJECT_ID=%~2

:: --- Path setup ---------------------------------------------
set SCRIPT_DIR=%~dp0
:: Remove trailing backslash
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

set INPUT_DIR=%SCRIPT_DIR%\input
set OUTPUT_DIR=%SCRIPT_DIR%\output
set LICENSE_DIR=%SCRIPT_DIR%\license

:: --- Pre-flight checks --------------------------------------
echo.
echo [INFO] Checking setup...

:: Check Docker
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop and try again.
    exit /b 1
)
echo [OK]   Docker is running.

:: Check license file
if not exist "%LICENSE_DIR%\license.txt" (
    echo [ERROR] FreeSurfer license not found at: %LICENSE_DIR%\license.txt
    echo         Register free at: https://surfer.nmr.mgh.harvard.edu/registration.html
    exit /b 1
)
echo [OK]   FreeSurfer license found.

:: Check input file
if not exist "%INPUT_DIR%\%INPUT_FILE%" (
    echo [ERROR] Input file not found: %INPUT_DIR%\%INPUT_FILE%
    exit /b 1
)
echo [OK]   Input file found: %INPUT_FILE%

:: Create output directory if needed
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: --- Run recon-all ------------------------------------------
echo.
echo [INFO] Starting recon-all for subject: %SUBJECT_ID%
echo [INFO] Using %NUM_CORES% CPU cores ^(-parallel -openmp %NUM_CORES%^)
echo [INFO] Expected runtime: ~5-7 hours ^(vs 6-12 hours single-core^)
echo [INFO] Tip: Set NUM_CORES at the top of this file to match your machine.
echo.

docker run --rm ^
    --cpus="%NUM_CORES%" ^
    -v "%INPUT_DIR%:/input" ^
    -v "%OUTPUT_DIR%:/output" ^
    -v "%LICENSE_DIR%:/license" ^
    -e FS_LICENSE=/license/license.txt ^
    -e OMP_NUM_THREADS=%NUM_CORES% ^
    freesurfer/freesurfer:7.4.1 ^
    recon-all ^
        -i /input/%INPUT_FILE% ^
        -s %SUBJECT_ID% ^
        -sd /output ^
        -all ^
        -parallel ^
        -openmp %NUM_CORES%

if errorlevel 1 (
    echo.
    echo [ERROR] recon-all failed. Check the log at:
    echo         %OUTPUT_DIR%\%SUBJECT_ID%\scripts\recon-all.log
    exit /b 1
)

echo.
echo [DONE] recon-all completed successfully.
echo [INFO] Results are in: %OUTPUT_DIR%\%SUBJECT_ID%\
echo.

endlocal
