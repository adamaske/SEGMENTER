@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  create_5tt.bat
::  Generates a merged 5-tissue-type (5TT) label volume from
::  a completed recon-all output.
::
::  Tissues and their output label values:
::    1 = White Matter  (WM)
::    2 = Grey Matter   (GM / cortex)
::    3 = CSF
::    4 = Skull
::    5 = Scalp
::
::  Usage:
::    create_5tt.bat <subject_id>
::
::  Prerequisites:
::    - recon-all must have completed for this subject
::    - Output must be in .\output\<subject_id>\
::    - FreeSurfer license.txt in .\license\license.txt
::    - Python + nibabel + numpy + scipy installed
:: ============================================================

:: --- Argument validation ------------------------------------
if "%~1"=="" (
    echo [ERROR] No subject ID specified.
    echo Usage: create_5tt.bat ^<subject_id^>
    exit /b 1
)

set SUBJECT_ID=%~1

:: --- Path setup ---------------------------------------------
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

set OUTPUT_DIR=%SCRIPT_DIR%\output
set LICENSE_DIR=%SCRIPT_DIR%\license
set SUBJECT_DIR=%OUTPUT_DIR%\%SUBJECT_ID%
set SCRIPTS_DIR=%SCRIPT_DIR%\scripts

:: --- Locate Python ------------------------------------------
:: Resolve the real .exe path upfront so spaces in user profile
:: directory don't break 'call' later. 'py' launcher is preferred
:: as it bypasses pyenv shims entirely.

for /f "delims=" %%i in ('where py 2^>nul') do (
    set PYTHON_EXE=%%i
    goto python_found
)
for /f "delims=" %%i in ('where python 2^>nul') do (
    set PYTHON_EXE=%%i
    goto python_found
)

echo [ERROR] Python not found. Install Python 3 and run:
echo         pip install nibabel numpy scipy
exit /b 1

:python_found
echo [OK]   Python: %PYTHON_EXE%

:: --- Pre-flight checks --------------------------------------
echo.
echo [INFO] Checking setup...

docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running.
    exit /b 1
)
echo [OK]   Docker is running.

if not exist "%LICENSE_DIR%\license.txt" (
    echo [ERROR] FreeSurfer license not found at: %LICENSE_DIR%\license.txt
    exit /b 1
)
echo [OK]   FreeSurfer license found.

if not exist "%SUBJECT_DIR%\mri\aseg.mgz" (
    echo [ERROR] recon-all output not found. Expected: %SUBJECT_DIR%\mri\aseg.mgz
    echo         Run run_recon_all.bat first.
    exit /b 1
)
echo [OK]   recon-all output found for subject: %SUBJECT_ID%

"%PYTHON_EXE%" -c "import nibabel, numpy, scipy" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Missing Python packages. Run:
    echo         "%PYTHON_EXE%" -m pip install nibabel numpy scipy
    exit /b 1
)
echo [OK]   Python packages found ^(nibabel, numpy, scipy^).

:: --- Step 1: Run mri_watershed to generate BEM surfaces -----
echo.
echo [INFO] Step 1/2 - Running mri_watershed to generate skull and scalp surfaces...
echo [INFO] This takes ~5-10 minutes.
echo.

docker run --rm ^
    -v "%OUTPUT_DIR%:/output" ^
    -v "%LICENSE_DIR%:/license" ^
    -e FS_LICENSE=/license/license.txt ^
    freesurfer/freesurfer:7.4.1 ^
    bash -c "mkdir -p /output/%SUBJECT_ID%/bem && mri_watershed -surf /output/%SUBJECT_ID%/bem/ws /output/%SUBJECT_ID%/mri/T1.mgz /output/%SUBJECT_ID%/mri/brainmask_ws.mgz"

if errorlevel 1 (
    echo [ERROR] mri_watershed failed. Check that T1.mgz exists in %SUBJECT_DIR%\mri\
    exit /b 1
)
echo [OK]   Watershed surfaces generated in %SUBJECT_DIR%\bem\

:: --- Step 2: Merge into 5TT volume via Python ----------------
echo.
echo [INFO] Step 2/2 - Merging aseg.mgz + watershed surfaces into 5TT label volume...
echo.

"%PYTHON_EXE%" "%SCRIPTS_DIR%\build_5tt.py" "%SUBJECT_DIR%"

if errorlevel 1 (
    echo [ERROR] 5TT merging failed. Check the error above.
    exit /b 1
)

echo.
echo [DONE] 5-tissue label volume created successfully.
echo [INFO] Output: %SUBJECT_DIR%\mri\5tt.mgz
echo.
echo [INFO] Label values:
echo         1 = White Matter
echo         2 = Grey Matter ^(cortex^)
echo         3 = CSF
echo         4 = Skull
echo         5 = Scalp
echo.

endlocal