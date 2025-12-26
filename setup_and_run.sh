#!/usr/bin/env bash
set -euo pipefail

# 一键环境初始化并启动服务脚本
# 功能：
# 1) 创建并激活 Conda 环境
# 2) 配置 pip 清华源
# 3) 安装项目依赖
# 4) 启动 FastAPI 服务并进行一次自测

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="fontclassifity"
PYTHON_VERSION="3.10"
PORT="8000"

echo "[1/6] 检查 Conda 可用性..."
if ! command -v conda >/dev/null 2>&1; then
  echo "未检测到 conda，请先安装 Miniconda/Anaconda 后重试。"
  echo "安装参考：https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

echo "[2/6] 初始化 conda shell..."
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
  echo "检测到已存在的环境 ${ENV_NAME}，跳过创建。"
else
  echo "创建 Conda 环境：${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

echo "激活环境：${ENV_NAME}"
conda activate "${ENV_NAME}"

echo "[3/6] 配置 pip 清华源..."
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple || true
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn || true

echo "[4/6] 安装项目依赖..."
cd "${PROJECT_DIR}"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[5/6] 检查端口占用并启动服务..."
# 若 8000 被占用，则尝试 8001-8010
for candidate in $(seq 8000 8010); do
  if ! lsof -i:"${candidate}" -sTCP:LISTEN >/dev/null 2>&1; then
    PORT="${candidate}"
    break
  fi
done
echo "使用端口：${PORT}"
nohup uvicorn main:app --host 0.0.0.0 --port "${PORT}" > "${PROJECT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
echo "服务已后台启动，PID=${SERVER_PID}，日志：${PROJECT_DIR}/server.log"

echo "等待服务健康检查..."
HEALTH_URL="http://localhost:${PORT}/health"
ATTEMPTS=0
until curl -s "${HEALTH_URL}" | grep -q '"healthy"'; do
  ATTEMPTS=$((ATTEMPTS+1))
  if [ "${ATTEMPTS}" -gt 20 ]; then
    echo "健康检查失败，查看日志：${PROJECT_DIR}/server.log"
    exit 1
  fi
  sleep 1
done
echo "健康检查通过。"

echo "[6/6] 进行一次自测预测..."
python "${PROJECT_DIR}/testCode.py" || true

echo "服务保持运行中。访问：http://localhost:${PORT}/"
echo "如需停止服务：kill ${SERVER_PID}"