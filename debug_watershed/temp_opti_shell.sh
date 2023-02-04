#!/bin/bash -l

#SBATCH
#SBATCH --time=24:00:00
#SBATCH --partition=lrgmem
#SBATCH --nodes=1


Rscript temp_opti.R