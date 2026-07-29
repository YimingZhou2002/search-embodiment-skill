#!/bin/bash
# Preflight GPU idle check for embodiment training runs.
#
# Exit 0 => GPUs look idle, safe to launch.
# Exit 1 => GPUs are busy (running compute processes or high utilization).
#
# "Busy" heuristic: any GPU compute process is running, OR max GPU utilization
# exceeds UTIL_THRESH (default 10%), OR max used memory exceeds MEM_THRESH_MB
# (default 2000 MiB). Override thresholds via env vars.

UTIL_THRESH=${UTIL_THRESH:-10}
MEM_THRESH_MB=${MEM_THRESH_MB:-2000}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu-check] nvidia-smi not found; cannot verify GPU state." >&2
    exit 1
fi

echo "[gpu-check] Per-GPU (index, mem.used MiB, util %):"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits

procs=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^\s*$/d')

busy=0
if [ -n "$procs" ]; then
    echo "[gpu-check] Active compute processes:"
    echo "$procs" | sed 's/^/  /'
    busy=1
fi

max_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1)
max_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "${max_util:-0}" -gt "$UTIL_THRESH" ] && busy=1
[ "${max_mem:-0}" -gt "$MEM_THRESH_MB" ] && busy=1

if [ "$busy" -eq 1 ]; then
    echo "[gpu-check] GPUs are BUSY (max util=${max_util}%, max mem=${max_mem}MiB, thresholds util>${UTIL_THRESH}% mem>${MEM_THRESH_MB}MiB). Refusing to launch." >&2
    exit 1
fi

echo "[gpu-check] GPUs look idle (max util=${max_util}%, max mem=${max_mem}MiB). OK to launch."
exit 0
