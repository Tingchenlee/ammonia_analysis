#!/bin/sh

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=5:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=ammonia_temp
#SBATCH --output=logs/ammonia_Rebrov_temp/ammonia_Rebrov_temp.%a.log
#SBATCH --error=logs/ammonia_Rebrov_temp/ammonia_Rebrov_temp.%a.slurm.log
#SBATCH --partition=short
#SBATCH --mail-user=lee.ting@northeastern.edu
#SBATCH --mail-type=FAIL,END

#an array for the job.
#SBATCH --array=1-29


####################################################
source activate rmg_env
python -u ammonia_PFR_reactor_script_temp.py