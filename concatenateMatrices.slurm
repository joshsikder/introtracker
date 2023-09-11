#!/bin/bash
#SBATCH --job-name=catMat
#SBATCH -N 1
#SBATCH -t 0-01:00:00
#SBATCH --mem=64g
#SBATCH -n 1
#SBATCH --mail-user=jsikder@ad.unc.edu 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/concatMat_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/concatMat_%A.err

for i in `seq 1 13`
    do
        for x in `seq 1 5500`
            do
                cat "/nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/supermatrices/simulation${i}_${x}.mat.fasta"
            done | awk '{{print}}' > /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_training_all_matrices.txt

        for y in `seq 5501 11000`
            do
                cat "/nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/supermatrices/simulation${i}_${y}.mat.fasta"
            done | awk '{{print}}' > /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_test_all_matrices.txt
    done

sbatch --dependency=afterok:${SLURM_JOB_ID} --export=ALL /nas/longleaf/home/jsikder/phylo1/workflow/scripts/convertToNumpy.slurm