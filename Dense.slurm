#!/bin/bash
#SBATCH -p dschridelab
#SBATCH --job-name=DenseTemp2_lr0001
#SBATCH -N 1
#SBATCH -t 0-00:30:00
#SBATCH --mem=32g 
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/DenseTemp2_lr0001.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/DenseTemp2_lr0001.err

source activate /nas/longleaf/home/jsikder/.conda/envs/josh

python /nas/longleaf/home/jsikder/introtracker/workflow/scripts/Dense.py \
    -cf /nas/longleaf/home/jsikder/introtracker/workflow/conf/Dense.yaml \
    -c temp2 \
    --run