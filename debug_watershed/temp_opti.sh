#!/bin/bash -l

#SBATCH
#SBATCH --time=24:00:00
#SBATCH --partition=lrgmem
#SBATCH --nodes=1

step_size="$1"
convergence_thresh="$2"

Rscript temp_opti.R $step_size $convergence_thresh
