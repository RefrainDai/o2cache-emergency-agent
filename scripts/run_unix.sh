#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "[错误] 未检测到 Python。请先安装 Python 3.10 或更高版本。"
  exit 1
fi

echo "[1/2] 正在安装或检查依赖..."
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "[2/2] 正在启动应急智枢平台..."
"$PYTHON_BIN" -m streamlit run app.py
