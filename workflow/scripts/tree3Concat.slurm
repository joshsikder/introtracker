#!/bin/bash
#SBATCH --job-name=tree3Concat
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=256g
#SBATCH -n 4
#SBATCH --mail-user=jsikder@ad.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/concatTree3_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/concatTree3_%A.err

python /nas/longleaf/home/jsikder/phylo1/workflow/scripts/numpyConcat.py \
        /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation7/preprocessing/simulation7_gsC_matrices.npy \
        /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation8/preprocessing/simulation8_gsC_matrices.npy \
        /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation9/preprocessing/simulation9_gsC_matrices.npy \
        --output /nas/longleaf/home/jsikder/phylo1/workflow/datasets/treeMatrices/tree3_matrices.npy

sbatch --dependency=afterok:${SLURM_JOB_ID} --export=ALL /nas/longleaf/home/jsikder/phylo1/workflow/scripts/tree4Concat.slurm