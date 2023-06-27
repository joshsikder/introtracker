#!/bin/bash
#SBATCH -N 1
#SBATCH -t 7-00:00:00
#SBATCH --mem=5g
#SBATCH -n 1
#SBATCH --mail-user=jsikder@ad.unc.edu 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/compress_sim12_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/compress_sim12_%A.err

cd /work/users/j/s/jsikder/datasets/simulation12/
tar -czvf /proj/dschridelab/jsikder/datasets/sim12.tar.gz simulations