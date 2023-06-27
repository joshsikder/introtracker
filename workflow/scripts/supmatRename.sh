for j in $(ls /proj/dschridelab/jsikder/test/sm_temp/$1)
    do
        cd /proj/dschridelab/jsikder/test/sm_temp/$1/$j
        cp SuperMatrix.fas ../../$2/$j.fasta
    done
