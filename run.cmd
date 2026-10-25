@echo off
REM run.cmd - launcher for this workspace. usage: .\run 01_infer.py
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONPATH="
set "PYTHONUTF8=1"
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HOME=%~dp0.cache\hf"
set "HF_HUB_DISABLE_SYMLINKS=1"
"D:\anaconda\envs\llm\python.exe" %*
