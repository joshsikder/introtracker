#!/bin/bash
#SBATCH --job-name=tree2Concat
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=256g
#SBATCH -n 4
#SBATCH --mail-user=jsikder@ad.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/concatTree2_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/concatTree2_%A.err

python /nas/longleaf/home/jsikder/phylo1/workflow/scripts/numpyConcat.py \
        /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation4/preprocessing/simulation4_gsC_matrices.npy \
        /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation5/preprocessing/simulation5_gsC_matrices.npy \
        /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation6/preprocessing/simulation6_gsC_matrices.npy \
        --output /nas/longleaf/home/jsikder/phylo1/workflow/datasets/treeMatrices/tree2_matrices.npy

sbatch --dependency=afterok:${SLURM_JOB_ID} --export=ALL /nas/longleaf/home/jsikder/phylo1/workflow/scripts/tree3Concat.slurm