#!/bin/bash
#
# single_thread.batch
#

#SBATCH --job-name=contrastive_experiments      # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # A single task (process)
#SBATCH --partition=almanach                            # GPU parition
#SBATCH --cpus-per-task=16             # Using a single CPU
#SBATCH --gres=gpu:rtx8000:1                            # Request 1 GPU
#SBATCH --output=%x_%j.log            # Standard output and error log. %x denotes the job name, %j the jobid.
#SBATCH --time=2-00:00:00               # Time limit hrs:min:sec

module purge

source /home/$USER/.bashrc
conda activate halvesting_contrastive
torchrun --standalone --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_ON_NODE contrastive_experiments.py
