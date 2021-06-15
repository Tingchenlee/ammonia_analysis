#!/bin/sh

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=1:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=ammonia
#SBATCH --output=logs/ammonia_Rebrov/ammonia_Rebrov.%a.log
#SBATCH --error=logs/ammonia_Rebrov/ammonia_Rebrov.%a.slurm.log
#SBATCH --partition=short
#SBATCH --mail-user=lee.ting@northeastern.edu
#SBATCH --mail-type=FAIL,END

#an array for the job.
#SBATCH --array=1-935


####################################################
source activate rmg_env
python -u ammonia_PFR_reactor_script.py