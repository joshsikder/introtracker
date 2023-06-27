for j in `seq 1 11000`
do
    python2.7 /nas/longleaf/home/jsikder/phylo1/workflow/scripts/geneStitcher.py -in /nas/longleaf/home/jsikder/phylo1/workflow/datasets/$1/simulations/$2_${j}_*.fasta -matout /nas/longleaf/home/jsikder/phylo1/workflow/datasets/$1/supermatrices/$1_$j.mat.fasta -partout /nas/longleaf/home/jsikder/phylo1/workflow/datasets/$1/partitions/$1_$j.mat.part
done