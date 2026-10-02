@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo  === 음식 쇼츠 자동 편집 ===
echo.

where python >nul 2>nul || (
  echo [!] 파이썬이 설치되어 있지 않습니다.
  echo     https://www.python.org/downloads/ 에서 설치하세요.
  echo     설치 첫 화면에서 "Add python.exe to PATH" 를 꼭 체크하세요.
  pause & exit /b 1
)
where ffmpeg >nul 2>nul || (
  echo [!] ffmpeg 가 설치되어 있지 않습니다.
  echo     명령 프롬프트에서  winget install ffmpeg  를 실행한 뒤 이 창을 닫고 다시 실행하세요.
  pause & exit /b 1
)

if "%~1"=="" (
  if not exist raw mkdir raw
  echo 영상 파일을 끌어다 이 파일(edit.bat) 위에 놓거나,
  echo 이 폴더의 raw 폴더에 영상을 넣고 다시 실행하세요.
  echo.
  dir /b raw 2>nul | findstr /r "." >nul || (
    echo [!] raw 폴더가 비어 있습니다.
    explorer raw
    pause & exit /b 1
  )
  python autoedit.py raw
) else (
  python autoedit.py %*
)
if errorlevel 1 ( pause & exit /b 1 )

for /f "delims=" %%d in ('dir /b /ad /o-d out_* 2^>nul') do ( explorer "%%d" & goto :done )
:done
echo.
echo 완료! 열린 폴더의 final.mp4 가 완성본, clips 폴더가 캡컷용 컷 파일입니다.
pause
