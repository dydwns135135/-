#!/bin/bash
# 맥용: 더블클릭하거나 터미널에서 ./edit.command [영상파일들...]
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "python3 가 필요합니다: https://www.python.org/downloads/"; read -p "Enter"; exit 1; }
command -v ffmpeg  >/dev/null || { echo "ffmpeg 가 필요합니다: brew install ffmpeg"; read -p "Enter"; exit 1; }
if [ $# -eq 0 ]; then
  mkdir -p raw
  ls raw | grep -q . || { echo "raw 폴더에 영상을 넣고 다시 실행하세요."; open raw; read -p "Enter"; exit 1; }
  python3 autoedit.py raw
else
  python3 autoedit.py "$@"
fi
open "$(ls -dt out_* | head -1)"
read -p "완료! Enter 를 누르면 닫힙니다"
