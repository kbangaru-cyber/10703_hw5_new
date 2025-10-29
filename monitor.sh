#!/bin/bash
# Monitor experiments in real-time

cd /home/adityaku/HW/10703_HW5_new

echo "=========================================="
echo "PETS Experiments Monitor"
echo "=========================================="
echo ""

echo "1. Running Python processes:"
ps aux | grep "[p]ython.*\(run\|train\).py" | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "TIME:", $10, "CMD:", $11, $12, $13}' || echo "  No experiments running"
echo ""

echo "2. GPU Usage:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | awk -F', ' '{print "  GPU: " $1 " | Memory: " $2 " / " $3 " | Temp: " $4}'
echo ""

echo "3. Log files:"
for log in q*.log *results*.log; do
  if [ -f "$log" ]; then
    size=$(ls -lh "$log" | awk '{print $5}')
    lines=$(wc -l < "$log")
    echo "  $log: $size ($lines lines)"
  fi
done
echo ""

echo "4. Generated plots:"
ls -1 out/*.png *.png 2>/dev/null | while read f; do
  size=$(ls -lh "$f" | awk '{print $5}')
  echo "  $f ($size)"
done || echo "  No plots generated yet"
echo ""

echo "5. Latest log output (last 20 lines):"
echo "----------------------------------------"
latest_log=$(ls -t q*.log *results*.log 2>/dev/null | head -1)
if [ -f "$latest_log" ]; then
  echo "From: $latest_log"
  echo ""
  tail -20 "$latest_log"
else
  echo "No log files found yet"
fi
echo ""

echo "=========================================="
echo "Commands:"
echo "  tail -f q1.1-1.2-1.3_results.log    # Watch Q1.1-1.3 progress"
echo "  tail -f q1.4_results.log             # Watch Q1.4 progress"
echo "  watch -n 2 nvidia-smi                # Monitor GPU"
echo "  ./monitor.sh                         # Run this monitor again"
echo "=========================================="
