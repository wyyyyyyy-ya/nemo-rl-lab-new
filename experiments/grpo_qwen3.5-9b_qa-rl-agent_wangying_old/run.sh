#!/usr/bin/env bash
# 多轮 QA Agent GRPO。入口：本目录 run.py（QAAgentEnv + qa_rl 数据集）。
# 用法： NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=h100 bash run.sh
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${EXP_DIR}/../../scripts/_run_experiment.sh" "${EXP_DIR}"
