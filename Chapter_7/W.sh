#!/bin/env bash
#SBATCH --array=0-99
#SBATCH --error="./out/W_script/%x-%j-%a.err"
#SBATCH --output="./out/W_script/%x-%j-%a.out"
#SBATCH --partition=ibIntel
#SBATCH --mem=5G
#SBATCH --time=02-00:00:00
#SBATCH --mail-type=END,FAIL

. ~/.bashrc
srun /marisdata/easybuild/software/QuantumMiniconda3/4.7.10/bin/python3 W_script.py 100 ${SLURM_ARRAY_TASK_ID}