@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0\.."

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未检测到 Python。请先安装 Python 3.10 或更高版本，并勾选 Add Python to PATH。
  pause
  exit /b 1
)

echo [1/2] 正在安装或检查依赖...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo [错误] 依赖安装失败，请检查网络或 Python 环境。
  pause
  exit /b 1
)

echo [2/2] 正在启动应急智枢平台...
python -m streamlit run app.py
pause
