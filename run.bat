@echo off
cd /d "%~dp0"
python -m streamlit run car_fault_ui.py
pause
