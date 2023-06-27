#!/bin/bash
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=8g
#SBATCH -n 1
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/supermat_sim2_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/supermat_sim2_%A.err

sh scripts/supermatrix.sh simulation2 sim2