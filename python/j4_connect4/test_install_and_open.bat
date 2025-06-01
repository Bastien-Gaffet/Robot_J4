@echo off
setlocal

echo ===============================
echo Checking for 'build' module
echo ===============================

python -c "import build" 2>NUL
if errorlevel 1 (
    echo 'build' module not found. Installing...
    pip install build
) else (
    echo 'build' module is already installed.
)

echo.
echo ===============================
echo Building the package
echo ===============================
python -m build

echo.
echo ===============================
echo Testing installation
echo ===============================

rem Create temporary virtual environment
echo [1/6] Creating virtual environment...
python -m venv test_env

rem Activate the virtual environment
call test_env\Scripts\activate.bat

rem Upgrade pip and install the built wheel
echo [2/6] Installing the built package...
python -m pip install --upgrade pip >nul
for /f %%f in ('dir /b dist\*.whl') do set PACKAGE=dist\%%f
pip install %PACKAGE%

rem Get the installed package path
echo [3/6] Locating installed package folder...
python -c "import connect4_robot_j4, os; print(os.path.dirname(connect4_robot_j4.__file__))" > tmp_path.txt
set /p PACKAGE_PATH=<tmp_path.txt
del tmp_path.txt

rem Open the installed package folder
echo [4/6] Opening installed package folder:
echo %PACKAGE_PATH%
explorer "%PACKAGE_PATH%"

rem Open the .dist-info folder
for /d %%d in ("%PACKAGE_PATH%\..\") do (
    for /d %%i in ("%%~fd\connect4_robot_j4-*.dist-info") do (
        echo [5/6] Opening %%~fi (.dist-info folder)
        explorer "%%~fi"
    )
)

rem Clean up temporary files and folders
echo.
echo ===============================
echo Cleaning up temporary files
echo ===============================
deactivate
rmdir /s /q test_env
rmdir /s /q build
rmdir /s /q connect4_robot_j4.egg-info

echo Done. Test completed successfully.
pause
endlocal