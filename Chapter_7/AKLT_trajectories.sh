#!/bin/env bash
#SBATCH --array=0-19
#SBATCH --error="./out/AKLT_trajectories/%x-%j-%a.err"
#SBATCH --output="./out/AKLT_trajectories/%x-%j-%a.out"
#SBATCH --partition=compIntel
#SBATCH --mem=5G
#SBATCH --time=02-00:00:00
#SBATCH --mail-type=END,FAIL

. ~/.bashrc
srun /marisdata/easybuild/software/QuantumMiniconda3/4.7.10/bin/python3 AKLT_trajectories_script.py ${SLURM_ARRAY_TASK_ID}