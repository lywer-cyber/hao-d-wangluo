#!/usr/bin/env bash
# 04 统一训练入口
#
# 所有模型走同一条命令、同一套数据、同一套超参，唯一变量是 --model。
# 这就是"所有实验都用的实验脚本"。
#
# 用法
#   ./04_train.sh DAT physical 4                  # 单个
#   ./04_train.sh --all                           # 全部模型 × 全部协议 × X4
#   ./04_train.sh --matrix 4                      # 生成论文对比矩阵
#
# 前置：
#   export BASICSR_ROOT=/home/zero/refs/DAT       # 你克隆的 DAT 仓库（自带 BasicSR）
#   把 RCAN / SwinIR / HAT / MambaIR 的 arch 文件放进 $BASICSR_ROOT/basicsr/archs/
#   并在 $BASICSR_ROOT/basicsr/archs/__init__.py 里 import，详见 README

set -euo pipefail

BASICSR_ROOT="${BASICSR_ROOT:-/home/zero/refs/DAT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJ_ROOT/configs"
LOG_DIR="$PROJ_ROOT/logs"
mkdir -p "$LOG_DIR"

MODELS=(SRResNet RCAN NAFNet SwinIR HAT DAT)
MODELS_MAMBA=(SRResNet RCAN NAFNet SwinIR HAT MambaIR DAT)
PROTOCOLS=(bicubic physical)

run_one() {
  local model="$1" protocol="$2" scale="$3"
  local cfg="$CONFIG_DIR/train/${model}_${protocol}_x${scale}.yml"
  local log="$LOG_DIR/${model}_${protocol}_x${scale}.log"

  if [[ ! -f "$cfg" ]]; then
    echo "!! 缺配置 $cfg，先跑 gen_configs.py"
    return 1
  fi

  echo "===== 训练 $model / $protocol / X$scale ====="
  echo "  配置: $cfg"
  echo "  日志: $log"
  cd "$BASICSR_ROOT"
  python -m basicsr.train -opt "$cfg" 2>&1 | tee "$log"
}

case "${1:-}" in
  --all)
    for m in "${MODELS[@]}"; do
      for p in "${PROTOCOLS[@]}"; do run_one "$m" "$p" 4; done
    done
    ;;
  --matrix)
    for m in "${MODELS_MAMBA[@]}"; do
      for p in "${PROTOCOLS[@]}"; do run_one "$m" "$p" "${2:-4}"; done
    done
    ;;
  --models)
    # ./04_train.sh --models DAT RCAN SwinIR --protocol physical --scale 4
    shift
    MODELS_ARG=(); while [[ "${1:-}" != "--protocol" && -n "${1:-}" ]]; do MODELS_ARG+=("$1"); shift; done
    shift  # 去掉 --protocol
    PROT="${1:-bicubic}"; shift || true
    SCALE="${2:-4}"
    for m in "${MODELS_ARG[@]}"; do run_one "$m" "$PROT" "$SCALE"; done
    ;;
  "")
    echo "用法:"
    echo "  ./04_train.sh DAT physical 4"
    echo "  ./04_train.sh --all"
    echo "  ./04_train.sh --matrix 4"
    exit 1
    ;;
  *)
    run_one "$1" "${2:-bicubic}" "${3:-4}"
    ;;
esac
