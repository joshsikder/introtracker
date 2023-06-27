#!/bin/bash
#SBATCH --job-name=npyConcat
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=128g
#SBATCH -n 13
#SBATCH --mail-user=jsikder@ad.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/concatNumpy_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/concatNumpy_%A.err

for i in `seq 1 13`
    do
        python /nas/longleaf/home/jsikder/phylo1/workflow/scripts/numpyConcat.py \
                /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_training_matrices.npy \
                /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_test_matrices.npy \
                --output /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_gsC_matrices.npy
    done

sbatch --dependency=afterok:${SLURM_JOB_ID} --export=ALL /nas/longleaf/home/jsikder/phylo1/workflow/scripts/tree1Concat.slurm