@echo off
REM hf.cmd - run the HuggingFace CLI with this project's env
REM usage: .\hf download <repo_id> --local-dir models\xxx
REM        .\hf cache list
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONPATH="
set "PYTHONUTF8=1"
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HOME=%~dp0.cache\hf"
set "HF_HUB_DISABLE_SYMLINKS=1"
"D:\anaconda\envs\llm\Scripts\hf.exe" %*
