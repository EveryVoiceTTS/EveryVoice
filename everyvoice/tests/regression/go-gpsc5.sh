#!/bin/bash

#SBATCH --job-name=EV-r-main
#SBATCH --partition=standard
#SBATCH --account=nrc_ict
#SBATCH --qos=low
#SBATCH --time=720
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --output=./%x.o%j
#SBATCH --error=./%x.e%j

export SUBMIT_COMMAND="sbatch --qos=low --partition=gpu_v100 --account=nrc_ict__gpu_v100"
bash go.sh
