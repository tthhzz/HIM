@echo off
setlocal

set "DATASET=data/locomo10.json"
set "MODEL=qwen2.5:3b"
set "BACKEND=ollama"
set "RATIO=0.1"
set "K=10"

if not exist "results\runs" mkdir "results\runs"

call :run him_full true true true
call :run him_without_encoding false false true
call :run him_without_consolidation true true false
exit /b %errorlevel%

:run
echo Running %~1
python experiments\evaluate_him.py --dataset "%DATASET%" --model "%MODEL%" --backend "%BACKEND%" --ratio "%RATIO%" --retrieve_k "%K%" --enable_source_monitoring "%~2" --enable_activation_retrieval true --enable_attention_gating "%~3" --enable_consolidation "%~4" --output "results/runs/%~1.json"
if errorlevel 1 exit /b %errorlevel%
exit /b 0
