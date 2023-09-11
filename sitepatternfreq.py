#!/usr/bin/env python3
from itertools import product, combinations, chain
import sys, argparse, os
import numpy as np


def get_site_patterns(array_in,comb):
    N_alns = array_in.shape[0]
    N_taxa = array_in.shape[1]
    translateDict={0:'A',1:'T',2:'C',3:'G',4:'-'}
    character_alphabet = list(map(str,np.unique(array_in)))
    character_alphabet = [item for item in character_alphabet if item != '-15']
    site_patterns = list(product(character_alphabet,repeat = comb))
    site_patterns = list(map('_'.join,site_patterns))
    print(f"Site Patterns: {site_patterns}")
    alph_patterns = []
    for i in site_patterns:
        newstring = []
        for j in i:
            if j != '_':
                newstring.append(translateDict[int(j)])
        alph_patterns.append(''.join(newstring))
    with open("/nas/longleaf/home/jsikder/phylo1/workflow/datasets/test/sitepatternlist.txt","w+") as outfile:
        outfile.writelines(i+'\n' for i in alph_patterns)
    taxa_list = list(map(str,list(range(0,N_taxa))))
    taxa_list = list(combinations(taxa_list,comb))
    taxa_list = list(map('_'.join,taxa_list))
    print(f"Taxa List: {taxa_list}")
    
    dtype =[tuple([i,"f8"]) for i in list(chain(*[['_'.join([x,y]) for y in site_patterns] for x in taxa_list]))]
    print(f"Constructed datatype: {dtype}")
    struc_array = np.zeros(N_alns, dtype=dtype)
 
    taxa_groups = [list(x) for x in list(combinations(list(range(N_taxa)),comb))]
    
    for i in range(N_alns):
        # print(f"1: {array_in.shape}")
        aln = array_in[i,:,:]
        # print(f"2: {aln.shape}")
        aln = aln[:,(aln != -15).any(axis=0)]
        # print(f"3: {aln.shape}")
        Aln_length = aln.shape[1]
        for s in range(Aln_length):
            site = aln[:,s]
            newSite = newSites(site)
            print(f"New Site: {newSite}")
            for group in taxa_groups:
                array_key = '_'.join(map(str,group+newSite[group].tolist()))
                struc_array[array_key][i]+=1/Aln_length
    return(struc_array.view((float,len(struc_array.dtype.names))))

def newSites(inputsite):
    nucleotides = {}
    newSite = []
    for i in inputsite:
        if not i in nucleotides:
            nucleotides[i]=len(nucleotides)
        newSite.append(nucleotides[i])
    return np.array(newSite)


def main():
    parser = argparse.ArgumentParser(description='numeric2pattern conversion Ready for Keras')
    parser.add_argument( '--tr', help = "Train dataset in NUMPY",dest='TRAIN',default="TRAIN.npy")
    parser.add_argument( '--te', help = "Test dataset in NUMPY",dest='TEST',default="TEST.npy")
    parser.add_argument( '--trout', help = "Training output",dest='TROUT',type=str)
    parser.add_argument( '--teout', help = "Test output",dest='TEOUT',type=str)
    parser.add_argument( '--co', help = "Taxa combinations",dest='COMB',type=int)
    args = parser.parse_args()

    print(f"Reading input {args.TRAIN} and {args.TEST}")
    test_data = np.load(args.TEST)
    print(f"Finished reading {args.TEST}")
    train_data = np.load(args.TRAIN)
    print(f"Finished reading {args.TRAIN}")
    test_data_site = get_site_patterns(test_data,args.COMB)
    print(f"Saving {args.TEST}")
    np.save(args.TEOUT, test_data_site)
    print(f"Finished saving {args.TEST}")
    train_data_site = get_site_patterns(train_data,args.COMB)
    print(f"Saving {args.TRAIN}")
    np.save(args.TROUT, train_data_site)
    print(f"Finished saving {args.TRAIN}")
    
if __name__ == "__main__":
    main()
