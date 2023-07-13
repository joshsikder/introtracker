#!/bin/bash
#SBATCH --job-name=rfSPF
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --mem=128g
#SBATCH -n 16
#SBATCH --mail-user=jsikder@ad.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/rf_classify_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/rf_classify_%A.err

python scripts/rfclassifier.py -i datasets/rfData/tree1.npz
python scripts/rfclassifier.py -i datasets/rfData/tree2.npz
python scripts/rfclassifier.py -i datasets/rfData/tree3.npz
python scripts/rfclassifier.py -i datasets/rfData/tree4.npz