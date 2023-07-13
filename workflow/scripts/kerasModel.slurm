#!/bin/bash
#SBATCH -p dschridelab 
#SBATCH --job-name=T4Keras
#SBATCH -N 1
#SBATCH -t 5-00:00:00
#SBATCH --mem=128g 
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=jsikder@ad.unc.edu 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/keras_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/keras_%A.err

source activate /nas/longleaf/home/jsikder/.conda/envs/josh

python /nas/longleaf/home/jsikder/phylo1/workflow/scripts/kerasModel.py -bs 1 -intro tree4 -out /nas/home/users/jsikder/introtracker/workflow/models