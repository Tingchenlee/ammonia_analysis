#!/bin/sh

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=1:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=ammonia_PFR
#SBATCH --output=logs/PFR.%a.log
#SBATCH --error=logs/PFR.%a.slurm.log
#SBATCH --partition=short

#an array for the job.
#SBATCH --array=1-84%50


####################################################
source activate rmg_env
python -u ammonia_PFR_reactor_script.py