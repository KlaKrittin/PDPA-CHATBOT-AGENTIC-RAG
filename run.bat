@echo off
echo Starting Streamlit App...
echo.

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found!
    echo Please create it first: python -m venv venv
    pause
    exit /b 1
)

REM Check and install required dependencies
echo Checking dependencies...
venv\Scripts\python.exe -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing streamlit and dependencies...
    venv\Scripts\python.exe -m pip install streamlit pyyaml python-dotenv pydantic openai qdrant-client langgraph sentence-transformers
)

REM Run streamlit app
venv\Scripts\python.exe -m streamlit run app_llama3.2.py

pause