#!/bin/bash
#SBATCH --job-name=confgen
#SBATCH --output=STDOUT
#SBATCH --output=STDERR
#SBATCH --partition=shared
#SBATCH --mail-type=fail
#SBATCH --mail-user=acaruso@ucsd.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH -t 48:00:00


module load netcdf
module load gsl
module load python

python3 source/act_learn.py > stdout 2> stderr
