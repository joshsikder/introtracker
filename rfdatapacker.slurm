#!/bin/bash
#SBATCH --job-name=rfdatapacker
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=32g
#SBATCH -n 4
#SBATCH --mail-user=jsikder@ad.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/concatTree1_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/concatTree1_%A.err

python scripts/rfdatapacker.py -i datasets/spfs/simulation[0-13]*.npy -l datasets/simulation[0-13]*/simulation[0-13]*.labels -o datasets/rfData/allSPF.npz
