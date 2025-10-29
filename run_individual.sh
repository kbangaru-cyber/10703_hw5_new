#!/bin/bash
# Run individual questions separately

cd /home/adityaku/HW/10703_HW5_new
mkdir -p out

case "$1" in
  q1.1)
    echo "Running Q1.1: CEM with ground truth dynamics"
    conda run -n rl_env python -c "from run import test_cem_gt_dynamics; test_cem_gt_dynamics(50)" 2>&1 | tee q1.1_results.log
    ;;
  q1.2)
    echo "Running Q1.2: Single network training"
    conda run -n rl_env python -c "import torch; from run import train_single_dynamics; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); train_single_dynamics(50, device=device)" 2>&1 | tee q1.2_results.log
    ;;
  q1.3)
    echo "Running Q1.3: PETS training"
    conda run -n rl_env python -c "import torch; from run import train_pets; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); train_pets(device=device)" 2>&1 | tee q1.3_results.log
    ;;
  q1.4)
    echo "Running Q1.4: TD3 with synthetic data"
    conda run -n rl_env python train.py 2>&1 | tee q1.4_results.log
    ;;
  all)
    echo "Running Q1.1, Q1.2, Q1.3 together"
    conda run -n rl_env python run.py 2>&1 | tee q1.1-1.2-1.3_results.log
    echo ""
    echo "Running Q1.4"
    conda run -n rl_env python train.py 2>&1 | tee q1.4_results.log
    ;;
  *)
    echo "Usage: $0 {q1.1|q1.2|q1.3|q1.4|all}"
    echo ""
    echo "Examples:"
    echo "  ./run_individual.sh q1.1     # Run only Q1.1"
    echo "  ./run_individual.sh q1.2     # Run only Q1.2"
    echo "  ./run_individual.sh all      # Run all questions"
    exit 1
    ;;
esac

echo ""
echo "Done! Check the log files and out/ directory for results."
