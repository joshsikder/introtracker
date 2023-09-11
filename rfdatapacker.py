import numpy as np
import argparse
#python scripts/rfdatapacker.py -i datasets/spfs/simulation[0-9]*.npy -l datasets/simulation[0-9]*/simulation[0-9]*.labels -o temp/rf.test.npy

def concatenateFiles(dictData, dictLabels, out):
    X = np.concatenate(list(dictData.values()))
    y = np.concatenate(list(dictLabels.values()))
    
    np.savez(f"{out}", **{'X': X, 'y': y})


def main():
    parser = argparse.ArgumentParser(description='Preprocessing for RF classifier. Will take data and label arrays as input and will output a single npz.')
    parser.add_argument( "-i", "--input", nargs='+', help = "List of input arrays. Must be at least 2 arrays.", dest='INPUT')
    parser.add_argument( "-l", "--labels", nargs='+', help = "List of label arrays. Must be at least 2 arrays.",dest='LABELS')
    parser.add_argument( "-o", "--output", help = "Output file location",dest='OUTPUT',required=True)
    args = parser.parse_args()

    my_arrays = {}
    for i, item in enumerate(args.INPUT):
        my_arrays[f"{i+1}"] = np.load(item)
    
    my_labels = {}
    for j, item in enumerate(args.LABELS):
        my_labels[f"{j+1}"] = np.loadtxt(item)

    concatenateFiles(my_arrays, my_labels, args.OUTPUT)

if __name__ == '__main__':
    main()