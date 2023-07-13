#!/bin/bash
#SBATCH --job-name=plotSPF
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --mem=32g
#SBATCH -n 16
#SBATCH --mail-user=jsikder@ad.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/plotSPF3d_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/plotSPF3d_%A.err

python /nas/longleaf/home/jsikder/introtracker/workflow/scripts/plotspf3d.py