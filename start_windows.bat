@echo off
:: 防止双击运行后闪退
if not defined RUN_IN_CMD (
    set "RUN_IN_CMD=1"
    start cmd /k "%~f0"
    exit /b
)

chcp 65001 >nul
title DragonJot - 点睛AI启动助手
setlocal enabledelayedexpansion


echo ==============================================
echo          DragonJot - 点睛AI启动助手
echo ==============================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

@REM :MAIN
@REM echo 您是否会配置Python环境？(Y/N)
@REM set /p "user_choice=请输入选择: "

@REM if /i "%user_choice%"=="Y" goto KNOW_SETUP
@REM if /i "%user_choice%"=="N" goto AUTO_SETUP

@REM echo.
@REM echo [错误] 无效输入，请输入 Y 或 N。
@REM pause
@REM cls
@REM goto MAIN

@REM :KNOW_SETUP
@REM echo.
@REM echo 请阅读 README.md 文件，按照说明配置好Python和依赖环境。
@REM echo.
@REM pause
@REM set /p "ready=是否已配置好环境？(Y/N): "
@REM if /i "%ready%"=="Y" (
@REM     goto RUN_APP
@REM ) else (
@REM     echo 请先完成配置后再运行本程序。
@REM     goto END
@REM )

:AUTO_SETUP
echo.
echo 检测Python环境...

if exist "%SCRIPT_DIR%python\python.exe" (
    echo 已检测到本地Python环境。
) else (
    echo 未检测到Python，将自动下载完整版本。
    echo 正在下载Python安装包，请稍候...
    powershell -Command "try { Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe' -OutFile 'python_installer.exe' -ErrorAction Stop } catch { exit 1 }"
    if not exist "python_installer.exe" (
        echo [错误] Python下载失败，请检查网络。
        goto END
    )

    echo 安装Python到当前目录...
    start /wait python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 TargetDir="%SCRIPT_DIR%python"
    del /f /q python_installer.exe >nul 2>&1

    if not exist "%SCRIPT_DIR%python\python.exe" (
        echo [错误] Python安装失败。
        goto END
    )
)

if not exist "%SCRIPT_DIR%venv\" (
    echo 创建虚拟环境...
    "%SCRIPT_DIR%python\python.exe" -m venv "%SCRIPT_DIR%venv"
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败。
        goto END
    )
)

echo 激活虚拟环境...
call "%SCRIPT_DIR%venv\Scripts\activate.bat"

echo 正在准备pip和依赖环境，请稍候...
python -m ensurepip --default-pip >nul 2>&1
python -m pip install --upgrade pip -q >nul 2>&1

if exist requirements.txt (
    echo 安装依赖中...（请勿关闭窗口）
    python -m pip install -r requirements.txt -q
    if errorlevel 1 (
        echo [错误] 依赖安装失败，显示详细日志：
        python -m pip install -r requirements.txt
        goto END
    )
) else (
    echo [错误] 未找到 requirements.txt 文件。
    goto END
)

:RUN_APP
echo.
echo ==============================================
echo 启动 DragonJot...
echo 浏览器将自动打开：http://localhost:8501
echo 按 Ctrl+C 可停止运行
echo ==============================================
echo.

if exist "%SCRIPT_DIR%venv\Scripts\streamlit.exe" (
    "%SCRIPT_DIR%venv\Scripts\streamlit.exe" run "%SCRIPT_DIR%app.py"
) else (
    streamlit run app.py
)

:END
echo.
echo ----------------------------------------------
echo 操作已完成或发生错误。
echo 请查看上方提示信息。
echo 按任意键关闭窗口...
echo ----------------------------------------------
pause >nul
exit /b
