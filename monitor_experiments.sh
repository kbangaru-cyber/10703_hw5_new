#!/bin/bash
echo "=== Monitoring PETS Experiments ==="
echo ""
echo "1. Current experiment progress:"
tail -30 full_experiment_results.txt | grep -E "(INFO|avg_|success|Test|Q1\.)"
echo ""
echo "2. GPU usage:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
echo ""
echo "3. Running Python processes:"
ps aux | grep "[p]ython.*run.py" | awk '{print "PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
echo ""
echo "4. Generated output files:"
ls -lh out/*.png 2>/dev/null || echo "No plots generated yet"
