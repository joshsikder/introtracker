#!/bin/bash
#SBATCH -p dschridelab
#SBATCH --job-name=DenseTemp4
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --mem=32g 
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=jsikder@ad.unc.edu 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/DenseTemp4.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/DenseTemp4.err

source activate /nas/longleaf/home/jsikder/.conda/envs/josh

python /nas/longleaf/home/jsikder/introtracker/workflow/scripts/Dense.py -cf /nas/longleaf/home/jsikder/introtracker/workflow/conf/Dense.yaml -c temp4 --run