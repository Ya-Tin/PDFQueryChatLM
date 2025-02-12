@echo off
cd /d "%~dp0"
if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
) else (
    echo Virtual environment not found. Please create one using: python -m venv .venv
    exit /b 1
)

echo Running Streamlit app...
.venv\Scripts\streamlit run app.py