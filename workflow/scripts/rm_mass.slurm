#!/bin/bash
#SBATCH -N 1
#SBATCH -t 7-00:00:00
#SBATCH --mem=4g 
#SBATCH -n 4
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/massdelete_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/massdelete_%A.err

#find /work/users/j/s/jsikder/newSims/tenthpercentsims/none_sims/ -maxdepth 1 -type f -name '*.fasta' -print0 | xargs  -0 -r cp -t "/nas/longleaf/home/jsikder/phylo1/workflow/data/simulations/case0/tenthpercent/"
#find /nas/longleaf/home/jsikder/phylo1/workflow/data/simulations/case0/ -maxdepth 1 -type f -name '*.fasta' -print0 | xargs  -0 -r cp -t "/nas/longleaf/home/jsikder/phylo1/workflow/data/simulations/case0/onepercent/"
# find /nas/longleaf/home/jsikder/phylo1/workflow/data/sm_temp/ -maxdepth 1 -type f -delete
# find /nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/simulations/ -maxdepth 1 -type f -name "*.out" -delete
# find /nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/simulations/ -maxdepth 1 -type f -name "*.error" -delete
# find /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation0/simulations/ -maxdepth 1 -type f -delete
# find /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation1/simulations/ -maxdepth 1 -type f -delete
# find /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation2/simulations/ -maxdepth 1 -type f -delete
# find /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation3/simulations/ -maxdepth 1 -type f -delete
# find /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation4/simulations/ -maxdepth 1 -type f -delete
# find /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation5/simulations/ -maxdepth 1 -type f -delete
# find /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation6/simulations/ -maxdepth 1 -type f -delete
# find /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation7/simulations/ -maxdepth 1 -type f -delete
# find /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation13/simulations/ -maxdepth 1 -type f -delete
find /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree1/ -maxdepth 1 -type f -delete
find /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree2/ -maxdepth 1 -type f -delete
find /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree3/ -maxdepth 1 -type f -delete
find /nas/longleaf/home/jsikder/phylo1/workflow/datasets/tree4/ -maxdepth 1 -type f -delete
# find /work/users/j/s/jsikder/newSims/ -type f -delete