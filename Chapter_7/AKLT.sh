#!/bin/env bash
#SBATCH --array=0-1
#SBATCH --error="./out/AKLT_script/%x-%j-%a.err"
#SBATCH --output="./out/AKLT_script/%x-%j-%a.out"
#SBATCH --partition=ibIntel
#SBATCH --mem=5G
#SBATCH --time=02-00:00:00
#SBATCH --mail-type=END,FAIL

. ~/.bashrc
srun /marisdata/easybuild/software/QuantumMiniconda3/4.7.10/bin/python3 AKLT_script.py 3 10 ${SLURM_ARRAY_TASK_ID}