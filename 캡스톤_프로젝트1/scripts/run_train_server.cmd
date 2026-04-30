@echo off
REM ============================================================
REM YOLOv26n-seg 학습 러너 (서버 실행용)
REM SSH_ONLY.md §4 운영 원칙: .cmd 러너 + 로그 분리 방식
REM ============================================================
REM 서버 SSH 실행 예:
REM   plink -P 2222 codextrain@192.168.0.63 ^
REM     "D:\claude_work\paper_reproduction\scripts\run_train_server.cmd"
REM ============================================================

set PROJECT=D:\claude_work\paper_reproduction
set PYEXE=C:\ProgramData\Anaconda3\envs\roadlcc-gpu\python.exe
set LOG_DIR=%PROJECT%\runs\logs
if not exist %LOG_DIR% mkdir %LOG_DIR%

set TS=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%
set TS=%TS: =0%
set LOG_FILE=%LOG_DIR%\train_%TS%.log

echo [%date% %time%] Training start > %LOG_FILE%
echo Project: %PROJECT% >> %LOG_FILE%
echo Python:  %PYEXE% >> %LOG_FILE%
echo. >> %LOG_FILE%

cd /d %PROJECT%
%PYEXE% scripts\02_train_yolo26n_seg.py >> %LOG_FILE% 2>&1
set EXITCODE=%ERRORLEVEL%

echo. >> %LOG_FILE%
echo [%date% %time%] Training end (exit %EXITCODE%) >> %LOG_FILE%

if %EXITCODE% EQU 0 (
    echo TRAIN_OK %LOG_FILE%
) else (
    echo TRAIN_FAIL %LOG_FILE%
)
exit /b %EXITCODE%
