@echo off
REM ============================================================
REM YOLO26n-seg 학습 러너 (서버 실행용)
REM 경로/계정은 환경에 맞게 수정하여 사용.
REM ============================================================

set PROJECT=%~dp0..
set PYEXE=python
set LOG_DIR=%PROJECT%\runs\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set TS=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%
set TS=%TS: =0%
set LOG_FILE=%LOG_DIR%\train_%TS%.log

echo [%date% %time%] Training start > "%LOG_FILE%"
echo Project: %PROJECT% >> "%LOG_FILE%"
echo Python:  %PYEXE% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

cd /d "%PROJECT%"
%PYEXE% scripts\01_train_yolo26n_seg.py --data configs\data_yolo26n.yaml >> "%LOG_FILE%" 2>&1
set EXITCODE=%ERRORLEVEL%

echo. >> "%LOG_FILE%"
echo [%date% %time%] Training end (exit %EXITCODE%) >> "%LOG_FILE%"

if %EXITCODE% EQU 0 (
    echo TRAIN_OK %LOG_FILE%
) else (
    echo TRAIN_FAIL %LOG_FILE%
)
exit /b %EXITCODE%
