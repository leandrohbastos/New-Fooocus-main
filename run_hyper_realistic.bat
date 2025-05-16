@echo off
echo Iniciando Fooocus - Versao Hiper-Realista...
echo.

REM Verifica se o Python está instalado
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python nao encontrado. Usando Python embutido...
    set PYTHON_CMD=.\python_embeded\python.exe
) else (
    set PYTHON_CMD=python
)

REM Verifica se está usando GPU NVIDIA
nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo GPU NVIDIA detectada. Iniciando com suporte a CUDA...
    %PYTHON_CMD% -s Fooocus\entry_with_update.py --preset hyper_realistic --always-high-vram
) else (
    echo GPU NVIDIA nao detectada. Iniciando em modo CPU...
    %PYTHON_CMD% -s Fooocus\entry_with_update.py --preset hyper_realistic --always-cpu
)

pause 