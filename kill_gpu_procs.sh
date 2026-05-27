#!/bin/bash
# Kill all processes holding GPU memory and clear CUDA context.
# Run this whenever you hit CUDA OOM from zombie/stale processes.

echo "=== Current GPU state ==="
nvidia-smi

echo ""
echo "=== Killing processes with active GPU memory usage ==="

# Method 1: kill via nvidia-smi listed PIDs (excludes PID 0 / kernel entries)
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -v '^$')
if [ -n "$GPU_PIDS" ]; then
    for pid in $GPU_PIDS; do
        echo "Killing GPU process PID $pid: $(ps -p $pid -o comm= 2>/dev/null || echo '(already gone)')"
        kill -9 "$pid" 2>/dev/null || true
    done
else
    echo "No active GPU compute processes found via nvidia-smi."
fi

# Method 2: kill any process holding /dev/nvidia* device files
FUSER_PIDS=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | grep -v '^$' | sort -u)
if [ -n "$FUSER_PIDS" ]; then
    for pid in $FUSER_PIDS; do
        echo "Killing /dev/nvidia* holder PID $pid: $(ps -p $pid -o comm= 2>/dev/null || echo '(already gone)')"
        kill -9 "$pid" 2>/dev/null || true
    done
else
    echo "No processes holding /dev/nvidia* device files."
fi

# Method 3: kill stale Python/torch processes (conservative - only defunct/zombie ones)
echo ""
echo "=== Checking for zombie Python processes ==="
ZOMBIE_PIDS=$(ps aux | awk '$8 == "Z" && ($11 ~ /python/ || $11 ~ /torch/) {print $2}')
if [ -n "$ZOMBIE_PIDS" ]; then
    for pid in $ZOMBIE_PIDS; do
        echo "Killing zombie Python PID $pid"
        kill -9 "$pid" 2>/dev/null || true
    done
else
    echo "No zombie Python processes found."
fi

# Method 4: kill ALL Python processes that have GPU memory allocated (nuclear option)
echo ""
echo "=== Killing ALL Python processes holding GPU memory (nuclear) ==="
ALL_GPU_PIDS=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | grep -v '^$' | sort -u)
if [ -n "$ALL_GPU_PIDS" ]; then
    for pid in $ALL_GPU_PIDS; do
        cmd=$(ps -p "$pid" -o comm= 2>/dev/null || echo "(gone)")
        echo "Killing PID $pid ($cmd)"
        kill -9 "$pid" 2>/dev/null || true
    done
else
    # Fallback: pkill any python using significant GPU RAM
    echo "Trying pkill python3 / python as fallback..."
    pkill -9 -f "main.py" 2>/dev/null || true
fi

echo ""
echo "=== Waiting 2 seconds for cleanup... ==="
sleep 2

echo ""
echo "=== GPU state after cleanup ==="
nvidia-smi

echo ""
echo "Done. CUDA memory should be free."
