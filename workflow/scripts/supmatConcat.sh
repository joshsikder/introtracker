for i in renamedA renamedB renamedNone
    do
        cd /proj/dschridelab/jsikder/test/sm_temp/$i/
        awk '{print}' *.fasta > ../ALL_MATRICES_$i\.txt
    done