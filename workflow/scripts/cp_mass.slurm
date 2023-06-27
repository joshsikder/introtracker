#!/bin/bash
#SBATCH -N 1
#SBATCH -t 7-00:00:00
#SBATCH --mem=1g
#SBATCH -n 1
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/masscopy_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/masscopy_%A.err

#cp /work/users/j/s/jsikder/newSims/onepercentsims/none_sims/*.fasta /nas/longleaf/home/jsikder/phylo1/workflow/data/simulations/none/onepercent/
#cp /work/users/j/s/jsikder/newSims/tenthpercentsims/none_sims/*.fasta /nas/longleaf/home/jsikder/phylo1/workflow/data/simulations/none/tenthpercent/

find /work/users/j/s/jsikder/newSims/onepercentsims/none_sims/ -name "*.fasta" -exec mv '{}' /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation0/simulations/ \;

# find /work/users/j/s/jsikder/newSims/onepercentsims/a_sims/ -name "*.fasta" -exec mv '{}' /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation1/simulations/ \;

# find /work/users/j/s/jsikder/newSims/onepercentsims/b_sims/ -name "*.fasta" -exec mv '{}' /nas/longleaf/home/jsikder/phylo1/workflow/datasets/simulation2/simulations/ \;