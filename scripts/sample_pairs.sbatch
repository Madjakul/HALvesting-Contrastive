#!/bin/sh

#SBATCH --job-name=halvesting_contrastive_sample    # Job name
#SBATCH --partition=almanach
#SBATCH --gres=gpu:rtx8000:1                            # Request 1 GPU
#SBATCH --nodes=1                                       # Run all processes on a single node
#SBATCH --ntasks=1                                      # Run a single task
#SBATCH --cpus-per-task=16                              # Number of CPU cores per task
#SBATCH --time=2-00:00:00                               # Time limit hrs:min:sec
#SBATCH --output=logs/%x_%j.log                         # Standard output and error log. %x denotes the job name, %j the jobid.

module purge

source /home/$USER/.bashrc
conda activate halvesting_contrastive

mkdir logs || true

./scripts/sample_pairs.sh
