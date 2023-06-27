#! /bin/bash

#directories
slurm_output_dir="/nas/longleaf/home/jsikder/schriderlab/ignored/slurm_output"
data_dir="/nas/longleaf/home/jsikder/phylo1/workflow/datasets"
scripts_dir="/nas/longleaf/home/jsikder/phylo1/workflow/scripts"

# order of execution
: '
1. concatenateMatrices.slurm - Concatenates genestitcher matrices which run after the \
   simulations complete

2. convertToNumpy.slurm - Converts step 1 concatenated matrices into numpy arrays, the format \
                          we use for Keras. This is the step we process half of the dataset as \
                          training and test respectively

3A. geneStitcherConcat.slurm - Uses numpyConcat to concatenate gene stitcher matrices from \
                               step 2 together into a single dataset once again

3B. sitePatternFreq.slurm - uses step 2 matrices to calculate site pattern frequencies, which \
                            scans along the simulated chromosome calculating essentially \
                            ABBA-BABA frequencies

4A. tree1Concat.slurm > tree2Concat.slurm > tree3Concat.slurm > tree4Concat.slurm - for each \
    family of introgression scenarios (tree1 = sim0, sim1, sim2, sim3), concatenates \ 
    all of the matrices from step 3A

4B. spfConcat.slurm - concatenates all of the site pattern frequency matrices into one per \
                      simulation family

5A. tree1_split.slurm > tree2_split.slurm > tree3_split.slurm > tree4_split.slurm - stratified \
                       split the data into single files for use with the keras data generator

5B. spfVis.slurm - creates a heatmap of the site pattern frequencies from step 4B to show \
                   concentrations of site patterns per introgression scenario

TO-DO: combine step 4B matrices into trees (like 4A), then stratified split them with the \
       labels associated.

       create random forest classifier and possibly multi-layer perceptron for \
       site pattern frequencies 
'
# first job - concatenateMatrices
jid1=$(sbatch --job-name=catMat -N 1 -n 1 --mem=64g -t 0-01:00:00 --mail-user=jsikder@ad.unc.edu --mail-type=BEGIN,END,FAIL --output=${slurm_output_dir}/concatMat_%A.out --error=${slurm_output_dir}/concatMat_%A.err ${scripts_dir}/concatenateMatrices.slurm)
if [[ $jid1 =~ (job )([^,]*) ]]; then jid1="${BASH_REMATCH[2]}"; else echo "no match found"; fi;

# second job - convertToNumpy
jid2=$(sbatch --job-name=npyConv --dependency=afterany:$jid1 -N 1 -n 13 --mem=64g -t 0-03:30:00 --mail-user=jsikder@ad.unc.edu --mail-type=BEGIN,END,FAIL --output=${slurm_output_dir}/npyConv_%A.out --error=${slurm_output_dir}/npyConv_%A.err ${scripts_dir}/convertToNumpy.slurm)
if [[ $jid2 =~ (job )([^,]*) ]]; then jid2="${BASH_REMATCH[2]}"; else echo "no match found"; fi;

# third job - geneStitcherConcat
jid3=$(sbatch --job-name=gsC --dependency=afterany:$jid2 -N 1 -n 13 --mem=128g -t 4-00:00:00 --mail-user=jsikder@ad.unc.edu --mail-type=BEGIN,END,FAIL --output=${slurm_output_dir}/gsC_%A.out --error=${slurm_output_dir}/gsC_%A.err ${scripts_dir}/geneStitcherConcat.slurm)
if [[ $jid3 =~ (job )([^,]*) ]]; then jid3="${BASH_REMATCH[2]}"; else echo "no match found"; fi;

# fourth job - treeXConcat (ALL JOBS SUBMIT AFTER 3 CONCURRENTLY)
jid4=$(sbatch --job-name=tree1Concat --dependency=afterany:$jid3 -N 1 -n 13 --mem=256g -t 0-02:00:00 --mail-user=jsikder@ad.unc.edu --mail-type=BEGIN,END,FAIL --output=${slurm_output_dir}/tree1Concat_%A.out --error=${slurm_output_dir}/tree1Concat_%A.err ${scripts_dir}/tree1Concat.slurm)
if [[ $jid4 =~ (job )([^,]*) ]]; then jid4="${BASH_REMATCH[2]}"; else echo "no match found"; fi;
jid5=$(sbatch --job-name=tree2Concat --dependency=afterany:$jid3 -N 1 -n 13 --mem=256g -t 0-02:00:00 --mail-user=jsikder@ad.unc.edu --mail-type=BEGIN,END,FAIL --output=${slurm_output_dir}/tree2Concat_%A.out --error=${slurm_output_dir}/tree2Concat_%A.err ${scripts_dir}/tree2Concat.slurm)
if [[ $jid5 =~ (job )([^,]*) ]]; then jid5="${BASH_REMATCH[2]}"; else echo "no match found"; fi;
jid6=$(sbatch --job-name=tree3Concat --dependency=afterany:$jid3 -N 1 -n 13 --mem=256g -t 0-02:00:00 --mail-user=jsikder@ad.unc.edu --mail-type=BEGIN,END,FAIL --output=${slurm_output_dir}/tree3Concat_%A.out --error=${slurm_output_dir}/tree3Concat_%A.err ${scripts_dir}/tree3Concat.slurm)
if [[ $jid6 =~ (job )([^,]*) ]]; then jid6="${BASH_REMATCH[2]}"; else echo "no match found"; fi;
jid7=$(sbatch --job-name=tree4Concat --dependency=afterany:$jid3 -N 1 -n 13 --mem=512g -t 0-02:00:00 --mail-user=jsikder@ad.unc.edu --mail-type=BEGIN,END,FAIL --output=${slurm_output_dir}/tree4Concat_%A.out --error=${slurm_output_dir}/tree4Concat_%A.err ${scripts_dir}/tree4Concat.slurm)
if [[ $jid7 =~ (job )([^,]*) ]]; then jid7="${BASH_REMATCH[2]}"; else echo "no match found"; fi;


# multiple jobs can depend on a single job
jid2=$(sbatch  --dependency=afterany:$jid1 --mem=20g job2.sh)
jid3=$(sbatch  --dependency=afterany:$jid1 --mem=20g job3.sh)

# a single job can depend on multiple jobs
jid4=$(sbatch  --dependency=afterany:$jid2:$jid3 job4.sh)

# swarm can use dependencies
jid5=$(swarm --dependency=afterany:$jid4 -t 4 -g 4 -f job5.sh)

# a single job can depend on an array job
# it will start executing when all arrayjobs have finished
jid6=$(sbatch --dependency=afterany:$jid5 job6.sh)

# a single job can depend on all jobs by the same user with the same name
jid7=$(sbatch --dependency=afterany:$jid6 --job-name=dtest job7.sh)
jid8=$(sbatch --dependency=afterany:$jid6 --job-name=dtest job8.sh)
sbatch --dependency=singleton --job-name=dtest job9.sh