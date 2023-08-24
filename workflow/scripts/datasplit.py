import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from genutils import GeneralUtilities as gu
import argparse



def parseConfig(cfgFile, cfg):
    configurations = gu.readDatasetsConfig(cfgFile)
    return configurations[cfg]
def main():
    parser = argparse.ArgumentParser(description='Splits data to be passed as input to the data generator for Keras')
    parser.add_argument( '--config-file', '-cf', help = "Configuration filepath",dest='configfile', required = True, type=str)
    parser.add_argument( '--config', '-c', help = "Set of configurations",dest='config', required = True, type=str)
    args = parser.parse_args()
    
    cfgFile = args.configfile
    cfg = args.config
    # print(f'{cfg}: {parseConfig(cfgFile, cfg)}')
    dataset = parseConfig(cfgFile, cfg)
    dataPath = dataset['data']
    labelsPath = dataset['labels']
    batchsize = dataset['batch_size']
    numOutput = dataset['num_output_files']
    shuffle = dataset['shuffle']
    outputPath = dataset['output_dir']
    outputPath2 = dataset['output_dir_2']

    print(f"Loading data: {dataPath} and {labelsPath}")

    X = np.load(dataPath, mmap_mode="r")
    y = np.loadtxt(labelsPath)

    train_idxs, test_idxs, train_labs, test_labs = train_test_split(range(len(y)), y, test_size=(int(len(y) / 3)), stratify=y)  # Split into train/test and stratify by label but we only care about the indices now

    makeFiles(X, y, test_idxs, outputPath, outputPath2, batchsize, shuffle, 'test', numOutput)
    makeFiles(X, y, train_idxs, outputPath, outputPath2, batchsize, shuffle, 'train', numOutput)


def makeFiles(data, labels, idxs, path, path2, batch_size, shuffle, dataset, numOut):
    inds = idxs
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