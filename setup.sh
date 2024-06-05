#!/bin/bash

# 设置错误处理
set -e

# 更新包列表并安装必要的软件包
sudo apt-get update
sudo apt-get install -y wget bzip2

# 下载并安装Miniconda（如果尚未安装）
if ! command -v conda &> /dev/null
then
    echo "Conda 未找到，正在安装 Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
fi

# 创建并激活Conda虚拟环境
conda create -y -n opencv python=3.10
source $HOME/miniconda/bin/activate opencv

# 安装系统依赖
sudo apt-get update
sudo apt-get install -y libgtk2.0-dev libcanberra-gtk-module libgtk-3-dev

# 安装Python依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 提示用户已完成
echo "环境设置完成并已激活 opencv 虚拟环境"
echo "使用 'conda activate opencv' 激活该环境"

# （可选）运行项目的主程序
# python linux/main.py
