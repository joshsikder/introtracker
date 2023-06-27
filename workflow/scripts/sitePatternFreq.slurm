#!/bin/bash
#SBATCH --job-name=spf
#SBATCH -N 1
#SBATCH -t 0-04:00:00
#SBATCH --mem=128g
#SBATCH -n 5
#SBATCH --mail-user=jsikder@ad.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/spfMod_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/spfMod_%A.err

# for i in 0 1 2 3 7 8 9
#     do
#         echo "working on file $i"
#         python /nas/longleaf/home/jsikder/phylo1/workflow/scripts/sitepatternfreq.py \
#         --tr /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_training_matrices.npy \
#         --te /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_test_matrices.npy \
#         --trout /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_training_spf.npy \
#         --teout /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_test_spf.npy \
#         --co 3
#         echo "finished file $i"
#     done

# for i in 4 5 6 10 11 12 13
#     do
#         echo "working on file $i"
#         python /nas/longleaf/home/jsikder/phylo1/workflow/scripts/sitepatternfreq.py \
#         --tr /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_training_matrices.npy \
#         --te /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_test_matrices.npy \
#         --trout /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_training_spf.npy \
#         --teout /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_test_spf.npy \
#         --co 4
#         echo "finished file $i"
#     done

for i in 1 2
    do
        echo "working on file $i"
        python /nas/longleaf/home/jsikder/phylo1/workflow/scripts/sitepatternfreq.py \
        --tr /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_training_matrices.npy \
        --te /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation${i}/preprocessing/simulation${i}_test_matrices.npy \
        --trout /nas/longleaf/home/jsikder/phylo1/workflow/datasets/test/simulation${i}_training_spf.npy \
        --teout /nas/longleaf/home/jsikder/phylo1/workflow/datasets/test/simulation${i}_test_spf.npy \
        --co 3
        echo "finished file $i"
    done

# sbatch --dependency=afterok:${SLURM_JOB_ID} --export=ALL /nas/longleaf/home/jsikder/phylo1/workflow/scripts/spfConcat.slurm