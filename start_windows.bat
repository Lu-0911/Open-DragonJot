@echo off
chcp 65001 >nul 2>&1
echo ==============================================
echo          DragonJot - 点睛AI启动助手
echo ==============================================
echo.
echo 检测Python环境...

:: 检查Python是否已安装
where python >nul 2>&1
if %errorlevel% equ 0 (
    echo Python已安装
    python --version
) else (
    echo 未检测到Python环境，将自动安装Python 3.9...
    echo 正在下载Python安装包...
    
    :: 下载Python 3.9安装包
    if not exist "python_installer.exe" (
        powershell -Command "Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe -OutFile python_installer.exe"
    )
    
    :: 安装Python
    echo 正在安装Python...
    python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
    del python_installer.exe
    
    :: 刷新环境变量
    echo 安装完成，刷新环境变量...
    set "PATH=%PATH%;%APPDATA%\Python\Python39\Scripts;%USERPROFILE%\AppData\Local\Programs\Python\Python39\Scripts;%USERPROFILE%\AppData\Local\Programs\Python\Python39\"
)

:: 检查pip是否可用
echo 检查pip...
python -m ensurepip --default-pip >nul 2>&1
python -m pip install --upgrade pip >nul 2>&1


:: 安装依赖包
python -m pip install -r requirements.txt

:: 检查是否安装成功
if %errorlevel% equ 0 (
    echo 依赖安装完成，启动DragonJot...
    echo.
    echo ==============================================
    echo 系统将在浏览器中自动打开，若未打开请访问：
    echo http://localhost:8501
    echo 按Ctrl+C可停止程序
    echo ==============================================
    echo.
    streamlit run app.py
) else (
    echo 依赖安装失败，请检查网络连接后重试
    pause
)

