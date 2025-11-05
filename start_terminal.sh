#!/bin/bash
# ======================================================
# DragonJot - 点睛AI启动助手 (macOS/Linux 版)
# ======================================================
# 说明：
#  - 自动检测或安装 Python
#  - 自动创建并激活虚拟环境
#  - 自动安装依赖
#  - 启动 Streamlit 应用
# ======================================================

# 当前脚本目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "=============================================="
echo "          DragonJot - 点睛AI启动助手"
echo "=============================================="
echo

# -----------------------------------------------
# 检测Python环境
# -----------------------------------------------
echo "检测 Python 环境..."

if [ -x "$SCRIPT_DIR/python/bin/python3" ]; then
    echo "已检测到本地 Python 环境。"
    PYTHON_CMD="$SCRIPT_DIR/python/bin/python3"
else
    # 检查系统是否有 Python3
    if command -v python3 >/dev/null 2>&1; then
        echo "检测到系统 Python。"
        PYTHON_CMD="python3"
    else
        echo "未检测到 Python，将自动下载并安装 Python 3.9。"
        echo "正在下载 Python 安装包，请稍候..."

        # 下载 Python 安装包（macOS pkg）
        curl -L -o python_installer.pkg "https://www.python.org/ftp/python/3.9.13/python-3.9.13-macosx10.9.pkg"

        if [ ! -f "python_installer.pkg" ]; then
            echo "[错误] Python 下载失败，请检查网络连接。"
            exit 1
        fi

        echo "正在安装 Python..."
        sudo installer -pkg python_installer.pkg -target /
        rm -f python_installer.pkg

        # 再次检查安装
        if command -v python3 >/dev/null 2>&1; then
            PYTHON_CMD="python3"
        else
            echo "[错误] Python 安装失败。"
            exit 1
        fi
    fi
fi

# -----------------------------------------------
# 创建虚拟环境
# -----------------------------------------------
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "创建虚拟环境..."
    "$PYTHON_CMD" -m venv "$SCRIPT_DIR/venv"
    if [ $? -ne 0 ]; then
        echo "[错误] 虚拟环境创建失败。"
        exit 1
    fi
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source "$SCRIPT_DIR/venv/bin/activate"

# -----------------------------------------------
# 准备 pip 并安装依赖
# -----------------------------------------------
echo "正在准备 pip 环境..."
python -m ensurepip --default-pip >/dev/null 2>&1
python -m pip install --upgrade pip -q >/dev/null 2>&1

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "安装依赖中...（请勿关闭终端）"
    python -m pip install -r "$SCRIPT_DIR/requirements.txt" -q
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败，显示详细日志："
        python -m pip install -r "$SCRIPT_DIR/requirements.txt"
        exit 1
    fi
else
    echo "[错误] 未找到 requirements.txt 文件。"
    exit 1
fi

# -----------------------------------------------
# 启动应用
# -----------------------------------------------
echo
echo "=============================================="
echo "启动 DragonJot..."
echo "浏览器将自动打开：http://localhost:8501"
echo "按 Ctrl+C 可停止运行"
echo "=============================================="
echo

if [ -x "$SCRIPT_DIR/venv/bin/streamlit" ]; then
    "$SCRIPT_DIR/venv/bin/streamlit" run "$SCRIPT_DIR/app.py"
else
    streamlit run "$SCRIPT_DIR/app.py"
fi

# -----------------------------------------------
# 结束提示
# -----------------------------------------------
echo
echo "----------------------------------------------"
echo "操作已完成或发生错误。"
echo "请查看上方提示信息。"
echo "按任意键关闭窗口..."
echo "----------------------------------------------"
read -n 1 -s -r -p ""
echo
exit 0
