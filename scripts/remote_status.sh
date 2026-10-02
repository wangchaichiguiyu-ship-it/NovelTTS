#!/usr/bin/env bash
# =============================================================================
# remote_status.sh —— 一键查看云端全书生成进度
# =============================================================================
# 用法（在本地 Mac 上）：
#     ./scripts/remote_status.sh              # 看一次
#     ./scripts/remote_status.sh --watch 60   # 每 60 秒刷新（Ctrl-C 退出）
#
# 它只做只读查询，不会干扰正在跑的生成任务。
# =============================================================================
set -uo pipefail

HOST="${NOVELTTS_HOST:-autodl}"
REMOTE_DIR="${NOVELTTS_DIR:-/root/autodl-tmp/NovelTTS}"
OUT_DIR="${NOVELTTS_OUT:-/root/autodl-tmp/NovelTTS-output}"
TOTAL_SEGMENTS="${NOVELTTS_TOTAL:-3220}"

WATCH=0
if [[ "${1:-}" == "--watch" ]]; then
    WATCH="${2:-60}"
fi

render() {
    ssh -o ConnectTimeout=15 -o BatchMode=yes "$HOST" "
cd '$REMOTE_DIR' 2>/dev/null || exit 1

ALIVE=\$(ps -eo pid,sid,cmd | grep -c '[n]oveltts run')
ETIME=\$(ps -eo etime,cmd | grep '[n]oveltts run' | head -1 | awk '{print \$1}')
SEGS=\$(find '$OUT_DIR/audio/segments' -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
SIZE=\$(du -sh '$OUT_DIR' 2>/dev/null | cut -f1)
FAILED=\$(grep -c '最终失败' run_full.log 2>/dev/null || echo 0)
DEGRADED=\$(grep -c '批处理失败' run_full.log 2>/dev/null || echo 0)
GPU=\$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null)
DISK=\$(df -h /root/autodl-tmp | tail -1 | awk '{print \$4\" 可用 / \"\$2}')

# 已完成章节
CHAPTERS=\$(grep -E 'synthesis \| 完成 ' run_full.log 2>/dev/null | sed 's/.*synthesis | //' | tail -4)

echo \"ALIVE=\$ALIVE\"
echo \"ETIME=\$ETIME\"
echo \"SEGS=\$SEGS\"
echo \"SIZE=\$SIZE\"
echo \"FAILED=\$FAILED\"
echo \"DEGRADED=\$DEGRADED\"
echo \"GPU=\$GPU\"
echo \"DISK=\$DISK\"
echo '--- CHAPTERS ---'
echo \"\$CHAPTERS\"
"
}

show() {
    local raw
    raw="$(render)" || { echo "❌ 无法连接 $HOST（检查实例是否开机、ssh 别名是否有效）"; return 1; }

    local alive etime segs size failed degraded gpu disk
    alive="$(sed -n 's/^ALIVE=//p' <<<"$raw")"
    etime="$(sed -n 's/^ETIME=//p' <<<"$raw")"
    segs="$(sed -n 's/^SEGS=//p' <<<"$raw")"
    size="$(sed -n 's/^SIZE=//p' <<<"$raw")"
    failed="$(sed -n 's/^FAILED=//p' <<<"$raw")"
    degraded="$(sed -n 's/^DEGRADED=//p' <<<"$raw")"
    gpu="$(sed -n 's/^GPU=//p' <<<"$raw")"
    disk="$(sed -n 's/^DISK=//p' <<<"$raw")"

    local pct=0
    if [[ "$segs" =~ ^[0-9]+$ ]] && (( TOTAL_SEGMENTS > 0 )); then
        pct=$(( segs * 100 / TOTAL_SEGMENTS ))
    fi

    echo "════════════════════════════════════════════════════════"
    if [[ "$alive" == "1" ]]; then
        echo "  ✅ 生成中    已运行 $etime"
    else
        echo "  ⏹  进程已结束"
    fi
    echo "════════════════════════════════════════════════════════"
    printf "  片段进度   %s / %s  (%d%%)\n" "$segs" "$TOTAL_SEGMENTS" "$pct"
    printf "  已写体积   %s\n" "$size"
    printf "  GPU        %s\n" "$gpu"
    printf "  数据盘     %s\n" "$disk"
    printf "  失败/降级  %s / %s\n" "$failed" "$degraded"
    echo "────────────────────────────────────────────────────────"
    echo "  最近完成的章节："
    sed -n '/--- CHAPTERS ---/,$p' <<<"$raw" | tail -n +2 | sed 's/^/    /'
    echo "════════════════════════════════════════════════════════"
    echo "  看实时日志： ssh $HOST 'tail -f $REMOTE_DIR/run_full.log'"
}

if (( WATCH > 0 )); then
    while true; do
        clear
        date "+%Y-%m-%d %H:%M:%S"
        show
        echo
        echo "（每 ${WATCH}s 刷新，Ctrl-C 退出）"
        sleep "$WATCH"
    done
else
    show
fi
