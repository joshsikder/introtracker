#!/bin/bash
#SBATCH --job-name=tree2Split
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=512g
#SBATCH -n 1
#SBATCH --mail-user=jsikder@ad.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/splitTree2_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/splitTree2_%A.err

conda activate josh
source activate josh

python /nas/longleaf/home/jsikder/phylo1/workflow/scripts/datasplit.py \
    --indata /work/users/j/s/jsikder/datasets/treeMatrices/tree2_matrices.npy \
    --inlabels /work/users/j/s/jsikder/datasets/treeLabels/tree2.txt \
    --batch 1 \
    --out /work/users/j/s/jsikder/datasets/tree2

# sbatch --dependency=afterok:${SLURM_JOB_ID} --export=ALL /nas/longleaf/home/jsikder/phylo1/workflow/scripts/tree3_split.slurm