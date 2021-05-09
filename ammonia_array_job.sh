#!/bin/sh

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=1:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=ammonia
#SBATCH --output=logs/ammonia_Rebrov.%a.log
#SBATCH --error=logs/ammonia_Rebrov.%a.slurm.log
#SBATCH --partition=short

#an array for the job.
#SBATCH --array=1-2


####################################################
source activate rmg_env
python -u ammonia_reactor_script.py