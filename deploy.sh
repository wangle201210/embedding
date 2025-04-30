#!/bin/bash

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11.0"

function version_lt() { test "$(echo -e "$1\n$2" | sort -V | head -n 1)" != "$2"; }

if version_lt "$PYTHON_VERSION" "$REQUIRED_VERSION"; then
    echo "错误: Python版本必须 >= ${REQUIRED_VERSION}，当前版本为 ${PYTHON_VERSION}"
    exit 1
fi

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "创建Python虚拟环境..."
    python3 -m venv .venv
fi

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 检查HuggingFace登录状态
if ! python3 -c "from huggingface_hub.hf_api import HfApi; HfApi().whoami()" > /dev/null 2>&1; then
    echo "错误: 请先登录HuggingFace"
    echo "使用以下命令登录:"
    echo "huggingface-cli login"
    exit 1
fi

# 检查模型下载
echo "验证模型下载..."
if ! python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-m3')" > /dev/null 2>&1; then
    echo "错误: 无法下载或访问BAAI/bge-m3模型，请检查网络连接和HuggingFace登录状态"
    exit 1
fi

# 启动服务
echo "启动文本向量化服务..."
python3 main.py