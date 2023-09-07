#!/bin/bash
#SBATCH -p dschridelab 
#SBATCH --job-name=T1KerasDenseSPF
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 5-00:00:00
#SBATCH --mem=8g 
#SBATCH --gres=gpu:1
#SBATCH --mail-user=jsikder@ad.unc.edu 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/kerasDense_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/kerasDense_%A.err

source activate /nas/longleaf/home/jsikder/.conda/envs/josh

python /nas/longleaf/home/jsikder/introtracker/workflow/scripts/DenseSPF.py -cf /nas/longleaf/home/jsikder/introtracker/workflow/conf/DenseSPF.yaml -c tree1