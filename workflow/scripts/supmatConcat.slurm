#!/bin/bash
#SBATCH -p general 
#SBATCH -N 1
#SBATCH -t 0-48:00:00
#SBATCH --mem=64g 
#SBATCH -n 1
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/supmatConcat_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/supmatConcat_%A.error

sh /nas/longleaf/home/jsikder/schriderlab/ignored/phylo1/supmatConcat.sh