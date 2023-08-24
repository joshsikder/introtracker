import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

def main():
    parser = argparse.ArgumentParser(description='Splits data to be passed as input to the data generator for Keras')
    parser.add_argument( '--indata', '-d', help = "Input data, stored under datasets/treeMatrices",dest='DATA')
    parser.add_argument( '--inlabels', '-l', help = "Input labels, stored under datasets/treeLabels",dest='LABELS')
    parser.add_argument( '--batch', '-bs', help = "Number of alignments per file",dest='BATCHSIZE')
    parser.add_argument( '--numOutput', '-no', help = "Number of output files",dest='numOutput',type=int, required = false, default = None)
    parser.add_argument( '--shuffle', '-sh', help = "Shuffle output columns in second index (120000)", dest = 'SHUFFLE', default = None, type=bool)
    parser.add_argument( '--out', '-o', help = "Output directory",dest='OUTPUT')
    parser.add_argument( '--out2', '-o2', help = "Output directory2",dest='OUTPUT2', required = False, default = None)
    args = parser.parse_args()

    print(f"Loading data: {args.DATA} and {args.LABELS}")

    X = np.load(args.DATA, mmap_mode="r")
    y = np.loadtxt(args.LABELS)

    batchsize = int(args.BATCHSIZE)
    train_idxs, test_idxs, train_labs, test_labs = train_test_split(range(len(y)), y, test_size=(int(len(y) / 3)), stratify=y)  # Split into train/test and stratify by label but we only care about the indices now

    makeFiles(X, y, test_idxs, args.OUTPUT, args.OUTPUT2, batchsize, args.SHUFFLE, 'test', args.numOutput)
    makeFiles(X, y, train_idxs, args.OUTPUT, args.OUTPUT2, batchsize, args.SHUFFLE, 'train', args.numOutput)


def makeFiles(data, labels, idxs, path, path2, batch_size, shuffle, dataset, numOut):
    inds = idxs
    # Populate
    ind_arrays = []
    if numOut:
        while len(inds) >= batch_size and len(ind_arrays) < numOut:
            _inds = np.random.choice(inds, batch_size, replace=False)
            ind_arrays.append(_inds)
            inds = np.setxor1d(inds, _inds)
    else:
        while len(inds) >= batch_size:
            _inds = np.random.choice(inds, batch_size, replace=False)
            ind_arrays.append(_inds)
            inds = np.setxor1d(inds, _inds)

    for idx, batch_idxs in tqdm(enumerate(ind_arrays), desc=f"Saving {dataset} files", total=len(ind_arrays)):
        batch_data = data[batch_idxs]
        batch_labs = labels[batch_idxs]
        if shuffle == True and path2:
            indices = np.random.permutation(batch_data.shape[2])
            batch_data_shuf = batch_data[:,:,indices,:]
            np.savez(f"{path2.rstrip('/')}/{idx}_{dataset}_data.npz", **{"X": batch_data_shuf, "y": batch_labs})
        np.savez(f"{path.rstrip('/')}/{idx}_{dataset}_data.npz", **{"X": batch_data, "y": batch_labs})
        
if __name__ == '__main__':
    main()