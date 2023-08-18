#!/bin/bash
#SBATCH -p dschridelab 
#SBATCH --job-name=T1KerasDenseSPF
#SBATCH -N 1
#SBATCH -t 5-00:00:00
#SBATCH --mem=128g 
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=jsikder@ad.unc.edu 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/kerasDenseSPFTree1_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/kerasDenseSPFTree1_%A.err

source activate /nas/longleaf/home/jsikder/.conda/envs/josh

python /nas/longleaf/home/jsikder/introtracker/workflow/scripts/KerasModelDenseSPF.py -a /nas/longleaf/home/jsikder/introtracker/workflow/datasets/rfData/tree1.npz -s tree1 -o /nas/longleaf/home/jsikder/introtracker/workflow/models