#!/bin/env bash
#SBATCH --array=0-0
#SBATCH --error="./out/MDN/%x-%j-%a.err"
#SBATCH --output="./out/MDN/%x-%j-%a.out"
#SBATCH --partition=compAMD,compIntel,ibIntel
#SBATCH --exclude=maris[064],maris[048]
#SBATCH --mem=10G
#SBATCH --time=02-00:00:00
#SBATCH --mail-type=END,FAIL

. ~/.bashrc
srun /home/herasymenko/python_virt_envs/qsimcirq/bin/python3 AKLT_script.py 3 100 ${SLURM_ARRAY_TASK_ID}