#!/bin/bash
#SBATCH -p dschridelab 
#SBATCH --job-name=T1Keras_fullset_yshuf
#SBATCH -N 1
#SBATCH -t 5-00:00:00
#SBATCH --mem=128g 
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=jsikder@ad.unc.edu 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/keras_output/keras_fullset_yshuf_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/keras_output/keras_fullset_yshuf_%A.err

conda activate /work/users/j/s/jsikder/.conda/envs/josh
source activate /work/users/j/s/jsikder/.conda/envs/josh

python /nas/longleaf/home/jsikder/phylo1/workflow/scripts/kerasModel.py -bs 1 -intro tree1 -out /nas/home/users/jsikder/introtracker/workflow/models