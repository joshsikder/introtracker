#!/bin/bash
#SBATCH -p general 
#SBATCH -N 1
#SBATCH -t 0-01:00:00
#SBATCH --mem=1g 
#SBATCH -n 1

dirname = test

mkdir /proj/dschridelab/jsikder/$dirname

mkdir /proj/dschridelab/jsikder/$dirname/fastas
mkdir /proj/dschridelab/jsikder/$dirname/fastas/a_fastas
mkdir /proj/dschridelab/jsikder/$dirname/fastas/b_fastas
mkdir /proj/dschridelab/jsikder/$dirname/fastas/no_intro_fastas

mkdir /proj/dschridelab/jsikder/$dirname/supermatrices

mkdir /proj/dschridelab/jsikder/$dirname/newicks/
mkdir /proj/dschridelab/jsikder/$dirname/newicks/a
mkdir /proj/dschridelab/jsikder/$dirname/newicks/b
mkdir /proj/dschridelab/jsikder/$dirname/newicks/none

mkdir /proj/dschridelab/jsikder/$dirname/labels

mkdir /proj/dschridelab/jsikder/$dirname/sm_temp
mkdir /proj/dschridelab/jsikder/$dirname/sm_temp/a_matrices
mkdir /proj/dschridelab/jsikder/$dirname/sm_temp/b_matrices
mkdir /proj/dschridelab/jsikder/$dirname/sm_temp/none_matrices
mkdir /proj/dschridelab/jsikder/$dirname/sm_temp/renamedA
mkdir /proj/dschridelab/jsikder/$dirname/sm_temp/renamedB
mkdir /proj/dschridelab/jsikder/$dirname/sm_temp/renamedNone

for i in a b none
    do
        for j in `seq 11000`
        do
            mkdir /proj/dschridelab/jsikder/$dirname/sm_temp/$i\_matrices/$i\_$j\_
        done
    done
