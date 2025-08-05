@echo off
echo Starting Data Analyst Agent API...
echo ======================================
echo.

REM Show current directory
echo Current Directory: %cd%
echo.

REM Check if virtual environment exists
if exist ".venv\Scripts\python.exe" (
    echo Virtual environment found
    echo Using: .venv\Scripts\python.exe
    echo.
    
    REM Start the server
    echo Starting server on port 8000...
    echo Your API will be available at:
    echo    • http://localhost:8000
    echo    • http://localhost:8000/health
    echo    • http://localhost:8000/docs
    echo.
    echo Press Ctrl+C to stop the server
    echo ======================================
    
    .\.venv\Scripts\python.exe main.py
) else (
    echo Virtual environment not found!
    echo Using system Python...
    echo.
    python main.py
)

echo.
echo Server stopped.
echo Press any key to close this window...
pause > null