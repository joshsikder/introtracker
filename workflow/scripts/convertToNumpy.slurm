#!/bin/bash
#SBATCH --job-name=npyConv
#SBATCH -N 1
#SBATCH -t 0-03:30:00
#SBATCH --mem=64g
#SBATCH -n 13
#SBATCH --mail-user=jsikder@ad.unc.edu 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/convNumpy_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/convNumpy_%A.err

for i in `seq 1 2`
    do
        python /nas/longleaf/home/jsikder/phylo1/workflow/scripts/numpyConverter.py \
                --tr /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_training_all_matrices.txt \
                --te /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_test_all_matrices.txt \
                --trout /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_training_matrices.npy \
                --teout /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_test_matrices.npy
    done

sbatch --dependency=afterok:${SLURM_JOB_ID} --export=ALL /nas/longleaf/home/jsikder/phylo1/workflow/scripts/sitePatternFreq.slurm
sbatch --dependency=afterok:${SLURM_JOB_ID} --export=ALL /nas/longleaf/home/jsikder/phylo1/workflow/scripts/geneStitcherConcat.slurm