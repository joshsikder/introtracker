#!/bin/bash
#SBATCH --job-name=tree4Concat
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=512g
#SBATCH -n 4
#SBATCH --mail-user=jsikder@ad.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/concatTree4_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/concatTree4_%A.err

python /nas/longleaf/home/jsikder/phylo1/workflow/scripts/numpyConcat.py \
        /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation10/preprocessing/simulation10_gsC_matrices.npy \
        /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation11/preprocessing/simulation11_gsC_matrices.npy \
        /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation12/preprocessing/simulation12_gsC_matrices.npy \
        /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation13/preprocessing/simulation13_gsC_matrices.npy \
        --output /nas/longleaf/home/jsikder/phylo1/workflow/datasets/treeMatrices/tree4_matrices.npy

# sbatch --dependency=afterok:${SLURM_JOB_ID} --export=ALL /nas/longleaf/home/jsikder/phylo1/workflow/scripts/tree1_split.slurm