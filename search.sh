#!/bin/bash
# BM25 PDF Search - Quick Launch Script
# 直接启动搜索程序（不运行 EMB 生成）

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 禁用 Qt 调试日志输出
export QT_LOGGING_RULES="*.debug=false;qt.qpa.*=false"
export QT_DEBUG_PLUGINS=0

# 查找 PyQt5 的插件目录（自动检测 Python 版本）
PYQT5_PLUGINS=$(find "${SCRIPT_DIR}/.venv/lib" -type d -path "*/PyQt5/Qt5/plugins" 2>/dev/null | head -1)

if [ -n "$PYQT5_PLUGINS" ]; then
    export QT_PLUGIN_PATH="$PYQT5_PLUGINS"

    # 自动配置 fcitx5 输入法插件（如果系统有）
    PYQT5_IM_DIR="${PYQT5_PLUGINS}/platforminputcontexts"
    SYSTEM_FCITX5_PLUGIN=$(find /usr/lib -name "libfcitx5platforminputcontextplugin.so" 2>/dev/null | head -1)

    if [ -n "$SYSTEM_FCITX5_PLUGIN" ] && [ -d "$PYQT5_IM_DIR" ]; then
        # 如果 PyQt5 插件目录中没有 fcitx5 插件，创建软链接
        if [ ! -f "${PYQT5_IM_DIR}/libfcitx5platforminputcontextplugin.so" ]; then
            ln -sf "$SYSTEM_FCITX5_PLUGIN" "${PYQT5_IM_DIR}/" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✓ 已自动配置 fcitx5 输入法插件"
            fi
        fi
    fi
fi

# 配置输入法支持 (Rime/Fcitx5/IBus)
# 检测并设置正确的输入法模块
if [ -z "$QT_IM_MODULE" ]; then
    # 优先检测 fcitx5
    if pgrep -x "fcitx5" > /dev/null 2>&1; then
        export QT_IM_MODULE=fcitx
        export GTK_IM_MODULE=fcitx
        export XMODIFIERS=@im=fcitx
        echo "✓ 检测到 fcitx5，已配置输入法环境变量"
    # 其次检测 fcitx
    elif pgrep -x "fcitx" > /dev/null 2>&1; then
        export QT_IM_MODULE=fcitx
        export GTK_IM_MODULE=fcitx
        export XMODIFIERS=@im=fcitx
        echo "✓ 检测到 fcitx，已配置输入法环境变量"
    # 最后检测 ibus
    elif pgrep -x "ibus-daemon" > /dev/null 2>&1; then
        export QT_IM_MODULE=ibus
        export GTK_IM_MODULE=ibus
        export XMODIFIERS=@im=ibus
        echo "✓ 检测到 ibus，已配置输入法环境变量"
    else
        echo "⚠ 未检测到运行中的输入法框架，可能无法使用中文输入"
    fi
fi

# 启动搜索程序
cd "$SCRIPT_DIR"
echo "🔍 启动 BM25 PDF 搜索程序..."
uv run python search_app.py
