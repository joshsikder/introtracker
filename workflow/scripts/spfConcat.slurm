#!/bin/bash
#SBATCH --job-name=spfConcat
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --mem=64g
#SBATCH -n 1
#SBATCH --mail-user=jsikder@ad.unc.edu 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/spfConcat_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/spfConcat_%A.err

for i in `seq 1 13`
    do
        python /nas/longleaf/home/jsikder/introtracker/workflow/scripts/numpyConcat.py \
        /nas/longleaf/home/jsikder/introtracker/workflow/datasets/simulation${i}/preprocessing/simulation${i}_training_spf.npy \
        /nas/longleaf/home/jsikder/introtracker/workflow/datasets/simulation${i}/preprocessing/simulation${i}_test_spf.npy \
        --output /nas/longleaf/home/jsikder/introtracker/workflow/datasets/spfs/simulation${i}_spf.npy
    done

sbatch --dependency=afterok:${SLURM_JOB_ID} --export=ALL /nas/longleaf/home/jsikder/introtracker/workflow/scripts/spfVis.slurm