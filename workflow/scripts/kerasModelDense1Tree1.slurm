#!/bin/bash
#SBATCH -p gpu 
#SBATCH --job-name=T1KerasDense1Tree1Flatten
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --mem=32g 
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --qos gpu_access
#SBATCH --mail-user=jsikder@ad.unc.edu 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/kerasDense1Tree1Flatten_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/kerasDense1Tree1Flatten_%A.err

source activate /nas/longleaf/home/jsikder/.conda/envs/josh

python /nas/longleaf/home/jsikder/introtracker/workflow/scripts/kerasModelDense1.py -bs 100 -s tree1 -o /nas/longleaf/home/jsikder/introtracker/workflow/models