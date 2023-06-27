#!/bin/bash
#SBATCH --job-name=tree1Concat
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=256g
#SBATCH -n 4
#SBATCH --mail-user=jsikder@ad.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/concatTree1_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/concatTree1_%A.err

python /nas/longleaf/home/jsikder/phylo1/workflow/scripts/numpyConcat.py \
        /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation0/preprocessing/simulation0_gsC_matrices.npy \
        /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation1/preprocessing/simulation1_gsC_matrices.npy \
        /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation2/preprocessing/simulation2_gsC_matrices.npy \
        /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation3/preprocessing/simulation3_gsC_matrices.npy \
        --output /nas/longleaf/home/jsikder/phylo1/workflow/datasets/treeMatrices/tree1_matrices.npy

sbatch --dependency=afterok:${SLURM_JOB_ID} --export=ALL /nas/longleaf/home/jsikder/phylo1/workflow/scripts/tree2Concat.slurm