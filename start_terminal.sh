#!/bin/bash

echo "=============================================="
echo "          DragonJot - 点睛AI启动助手"
echo "=============================================="
echo

# 检查Python是否已安装
if command -v python3 &> /dev/null; then
    echo "Python已安装"
    python3 --version
    PY_CMD="python3"
elif command -v python &> /dev/null; then
    echo "Python已安装"
    python --version
    PY_CMD="python"
else
    echo "未检测到Python环境，将自动安装Python..."
    
    # 根据系统选择安装方式
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if ! command -v brew &> /dev/null; then
            echo "安装Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        echo "通过Homebrew安装Python..."
        brew install python@3.9
    else
        # Linux
        if command -v apt &> /dev/null; then
            sudo apt update
            sudo apt install -y python3.9 python3-pip
        elif command -v yum &> /dev/null; then
            sudo yum install -y python3.9 python3-pip
        else
            echo "不支持的Linux发行版，请手动安装Python 3.9+"
            exit 1
        fi
    fi
    
    PY_CMD="python3"
fi

# 升级pip
echo "升级pip..."
$PY_CMD -m pip install --upgrade pip

# 安装依赖包
echo "安装项目依赖..."
$PY_CMD -m pip install -r requirements.txt

# 启动应用
if [ $? -eq 0 ]; then
    echo "依赖安装完成，启动DragonJot..."
    echo
    echo "=============================================="
    echo "系统将在浏览器中自动打开，若未打开请访问："
    echo "http://localhost:8501"
    echo "按Ctrl+C可停止程序"
    echo "=============================================="
    echo
    streamlit run app.py
else
    echo "依赖安装失败，请检查网络连接后重试"
    exit 1
fi