@echo off
:: Set window title
title Asistent Juridic - RAG (Gradio)

:: Go to project directory
cd /d C:\\Users/screciun/PycharmProjects/am_internship-2024/md_legislation_ai_project

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Start the app
echo Launching your legal assistant...
python scripts/app.py

pause
