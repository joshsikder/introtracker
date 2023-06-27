#!/bin/bash
#SBATCH --job-name=spfVis
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --mem=8g
#SBATCH -n 1
#SBATCH --mail-user=jsikder@ad.unc.edu 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/spfVis_%A.out
#SBATCH --error=/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output/spfVis_%A.err

conda activate visualization
source activate visualization

python /nas/longleaf/home/jsikder/phylo1/workflow/scripts/visSitePatterns.py \
--spf /nas/longleaf/home/jsikder/phylo1/workflow/datasets/spfs/ \
--output /nas/longleaf/home/jsikder/phylo1/workflow/reports/spf_heatmap.png