#!/bin/bash
# ============================================================
# RLVMR Module Integration Tests
# 运行 tests/rlvmr 下的所有测试
# ============================================================

set -e  # 遇到错误时退出

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "       RLVMR Module Integration Tests        "
echo "=============================================="
echo ""

# 获取脚本所在目录的上级目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Project Root: $PROJECT_ROOT"
echo ""

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 检查 pytest 是否安装
if ! command -v python -m pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found. Install with: pip install pytest${NC}"
    exit 1
fi

# 设置 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Running tests from: tests/rlvmr/"
echo "----------------------------------------------"
echo ""

# 运行所有 rlvmr 测试
echo -e "${YELLOW}[1/4] Running test_discriminator_reward.py${NC}"
python -m pytest tests/rlvmr/test_discriminator_reward.py -v --tb=short || true
echo ""

echo -e "${YELLOW}[2/4] Running test_core_hybrid.py${NC}"
python -m pytest tests/rlvmr/test_core_hybrid.py -v --tb=short || true
echo ""

echo -e "${YELLOW}[3/4] Running test_expert_trajectory.py${NC}"
python -m pytest tests/rlvmr/test_expert_trajectory.py -v --tb=short || true
echo ""

echo -e "${YELLOW}[4/4] Running test_integration_hybrid.py${NC}"
python -m pytest tests/rlvmr/test_integration_hybrid.py -v --tb=short || true
echo ""

echo "----------------------------------------------"
echo ""

# 汇总运行（收集所有结果）
echo -e "${GREEN}Running ALL tests with summary:${NC}"
python -m pytest tests/rlvmr/ -v --tb=short -q

echo ""
echo "=============================================="
echo "                 Tests Complete               "
echo "=============================================="
