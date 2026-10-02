#!/usr/bin/env bash
# =============================================================================
# deploy_remote.sh —— 把本地代码同步到远程 GPU 实例并安装
# =============================================================================
#
# 为什么需要这个脚本（而不是手敲 tar/scp）：
#
#   **它会排除 config.yaml 与 config.local.yaml。**
#   这是实际踩过的坑：用 tar 覆盖式更新代码时把 config.yaml 一起覆盖了，
#   云端改好的 model_path 被冲回 HuggingFace 仓库 id，而云端连不上 HF，
#   模型直接加载失败。
#
#   机器相关的配置必须留在远程的 config.local.yaml 里（不入库），
#   代码更新永远不该碰它。
#
# 用法::
#
#   ./scripts/deploy_remote.sh                  # 用默认值（AutoDL 别名 autodl）
#   ./scripts/deploy_remote.sh --host autodl --dir /root/autodl-tmp/NovelTTS
#   ./scripts/deploy_remote.sh --dry-run        # 只打包，不传输
#   ./scripts/deploy_remote.sh --tests          # 传完顺便在远程跑测试
#
# 前置条件：
#   * `ssh <host>` 已能免密登录
#   * `ssh <host> 'which python'` 能找到解释器（AutoDL 的 conda 不在
#     非交互式 PATH 里，所以脚本默认用绝对路径 /root/miniconda3/bin/python）
# =============================================================================

set -euo pipefail

HOST="autodl"
REMOTE_DIR="/root/autodl-tmp/NovelTTS"
REMOTE_PYTHON="/root/miniconda3/bin/python"
RUN_TESTS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)       HOST="$2";        shift 2 ;;
        --dir)        REMOTE_DIR="$2";  shift 2 ;;
        --python)     REMOTE_PYTHON="$2"; shift 2 ;;
        --tests)      RUN_TESTS=1;      shift ;;
        --dry-run)    DRY_RUN=1;        shift ;;
        -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "未知参数：$1（用 --help 查看用法）" >&2; exit 2 ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ARCHIVE="$(mktemp -t noveltts-deploy-XXXXXX).tar.gz"
trap 'rm -f "$ARCHIVE"' EXIT

# -----------------------------------------------------------------------------
# 1) 打包：只带代码/测试/文档，**排除一切机器相关内容**
# -----------------------------------------------------------------------------
# 排除清单及理由：
#   config.yaml        -> 远程可能已按机器改过；代码更新不该碰它（用默认值即可，
#                         真正机器相关的设置应写进 config.local.yaml）
#   config.local.yaml  -> 机器专属，绝不覆盖
#   output/            -> 产物，远程自己生成
#   .git/              -> 远程不需要版本历史（要的话直接在远程 git clone）
#   sources/           -> 小说原文有版权且体积大，单独传
#   assets/            -> 参考音频首次部署后就在远程了，避免每次重传 13MB
#   Qwen3-TTS*/        -> 模型权重
#   .remote-key/       -> 私钥，绝不外传
# -----------------------------------------------------------------------------
EXCLUDES=(
    --exclude=output
    --exclude=.git
    --exclude='Qwen3-TTS*'
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='.graphflow-cache'
    --exclude='.remote-key'
    --exclude='.DS_Store'
    --exclude='config.yaml'
    --exclude='config.local.yaml'
    --exclude='sources'
    --exclude='assets'
)

tar "${EXCLUDES[@]}" -czf "$ARCHIVE" \
    noveltts tests docs scripts pyproject.toml README.md run_tests.py .gitignore .gitattributes

echo "已打包：$(du -h "$ARCHIVE" | cut -f1)（已排除 config.yaml / config.local.yaml / sources / assets）"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "--dry-run：打包内容如下，未传输"
    tar -tzf "$ARCHIVE" | head -40
    exit 0
fi

# -----------------------------------------------------------------------------
# 2) 传输
# -----------------------------------------------------------------------------
echo "传输到 $HOST:$REMOTE_DIR ..."
scp -q "$ARCHIVE" "$HOST:/tmp/noveltts-deploy.tar.gz"

# -----------------------------------------------------------------------------
# 3) 远程解压 + 安装
# -----------------------------------------------------------------------------
ssh "$HOST" "
set -e
mkdir -p '$REMOTE_DIR'
tar -xzf /tmp/noveltts-deploy.tar.gz -C '$REMOTE_DIR' 2>/dev/null
rm -f /tmp/noveltts-deploy.tar.gz
echo '代码已更新'
if [ -f '$REMOTE_DIR/config.local.yaml' ]; then
    echo 'config.local.yaml 存在（未被覆盖）：'
    grep -E 'model_path|output_dir|device' '$REMOTE_DIR/config.local.yaml' || true
else
    echo '⚠️  没有 config.local.yaml —— 请创建它并写入本机路径，例如：'
    echo '    generation:'
    echo '      model_path: /root/models/Qwen3-TTS-12Hz-1.7B-Base'
    echo '      output_dir: /root/autodl-tmp/NovelTTS-output   # 放数据盘'
fi
"

# -----------------------------------------------------------------------------
# 4) 可选：远程跑测试
# -----------------------------------------------------------------------------
if [[ "$RUN_TESTS" == "1" ]]; then
    echo
    echo "在远程运行测试（不加载模型、不占 GPU）..."
    ssh "$HOST" "cd '$REMOTE_DIR' && '$REMOTE_PYTHON' run_tests.py 2>&1 | tail -12"
fi
