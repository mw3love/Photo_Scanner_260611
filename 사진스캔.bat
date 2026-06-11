@echo off
rem 이 배치 파일이 있는 폴더로 이동(한글/공백 경로 안전)
cd /d "%~dp0"
rem 콘솔 창 없이 통합 사진 스캔 GUI 실행 (1장이든 폴더든 이 하나로)
start "" "C:\Python314\pythonw.exe" -m scanner.app %*
