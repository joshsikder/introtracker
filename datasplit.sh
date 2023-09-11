#!/bin/bash
#SBATCH --job-name=datasplit1
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=160g
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mail-user=jsikder@ad.unc.edu 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/datasplit1_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/datasplit1_%A.err

source activate /nas/longleaf/home/jsikder/.conda/envs/josh
conda activate /nas/longleaf/home/jsikder/.conda/envs/josh
i=1
python /nas/longleaf/home/jsikder/introtracker/workflow/scripts/datasplit.py -d /nas/longleaf/home/jsikder/introtracker/workflow/datasets/treeMatrices/tree${i}_matrices.npy -l /nas/longleaf/home/jsikder/introtracker/workflow/datasets/treeLabels/tree${i}.txt -bs 1 -sh False -no 300 -o /work/users/j/s/jsikder/fordan/temp