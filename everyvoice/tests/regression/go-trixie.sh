#!/bin/bash

#SBATCH --job-name=EVr-main
#SBATCH --partition=TrixieMain
#SBATCH --time=720
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --output=./%x.o%j
#SBATCH --error=./%x.e%j

# Command to submit a GPU job:
export SUBMIT_COMMAND="sbatch --partition=TrixieMain"
bash go.sh
