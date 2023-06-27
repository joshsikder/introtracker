#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=reindex
#SBATCH -t 7-00:00:00
#SBATCH --mem=5g
#SBATCH -n 4
#SBATCH --mail-user=jsikder@ad.unc.edu 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/reindex_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/reindex_%A.err

for i in 4
    do 
        for j in `seq 0 29333`
            do 
                mv /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree${i}/${j}_train_data.npz /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree${i}/$(( $j + 1 ))_data.npz
                # mv /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree${i}/${j}_train_data.npz /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree${i}/${j+1}_data.npz
            done
    done

for i in 4
    do 
        for j in `seq 0 14665`
            do 
                mv /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree${i}/${j}_test_data.npz /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree${i}/$(( $j + 29335 ))_data.npz
                # mv /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree${i}/${j}_test_data.npz /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree${i}/${j+29335}_data.npz
            done
    done

# for i in 2 3
#     do 
#         for j in `seq 0 21999`
#             do 
#                 mv /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree${i}/${j}_train_data.npz /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree${i}/$(( $j + 1 ))_data.npz
#             done
#     done

# for i in 2 3
#     do 
#         for j in `seq 0 10999`
#             do 
#                 mv /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree${i}/${j}_test_data.npz /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree${i}/$(( $j + 22001 ))_data.npz
#             done
#     done