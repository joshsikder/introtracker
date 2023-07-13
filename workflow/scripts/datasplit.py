import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

def main():
    parser = argparse.ArgumentParser(description='Splits data to be passed as input to the data generator for Keras')
    parser.add_argument( '--indata', help = "Input data",dest='DATA')
    parser.add_argument( '--inlabels', help = "Input labels",dest='LABELS')
    parser.add_argument( '--batch', help = "Number of alignments per file",dest='BATCHSIZE')
    parser.add_argument( '--out', help = "Output directory",dest='OUTPUT')
    args = parser.parse_args()

    # print(f"Loading data: {args.DATA} and {args.LABELS}")

    X = np.load(args.DATA, mmap_mode="r")
    y = np.loadtxt(args.LABELS)
    batchsize = int(args.BATCHSIZE)
    train_idxs, test_idxs, train_labs, test_labs = train_test_split(range(len(y)), y, test_size=(int(len(y) / 3)), stratify=y)  # Split into train/test and stratify by label but we only care about the indices now

    makeFiles(X, y, test_idxs, args.OUTPUT, batchsize, 'test')
    makeFiles(X, y, train_idxs, args.OUTPUT, batchsize, 'train')


def makeFiles(data, labels, idxs, path, batch_size, dataset):
    inds = idxs
    # Populate
    ind_arrays = []
    while len(inds) >= batch_size:
        _inds = np.random.choice(inds, batch_size, replace=False)  # Get the selection of inds
        ind_arrays.append(_inds)
        inds = np.setxor1d(inds, _inds)

    for idx, batch_idxs in tqdm(enumerate(ind_arrays), desc="Saving files", total=len(ind_arrays)):
        batch_data = data[batch_idxs]
        batch_labs = labels[batch_idxs]
        np.savez(f"{path}/{idx}_{dataset}_data.npz", **{"X": batch_data, "y": batch_labs})
        
if __name__ == '__main__':
    main()