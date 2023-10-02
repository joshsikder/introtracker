import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.file_utils import GeneralUtilities as gu
import argparse

def remove_files_if_directory_not_empty(directory):
    if not os.path.isdir(directory):
        print(f"{directory} is not a directory")
        return

    if os.listdir(directory):
        print(f"{directory} is not empty, removing files...")
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        print(f"{directory} is empty")

def downsampleData(data, proportion):
    if proportion < 0 or proportion > 100:
        raise ValueError("Downsample percentage must be between 0 and 100")
    block_size = 1200
    num_blocks = int(data.shape[2] / block_size)
    num_blocks_to_remove = int(num_blocks * proportion / 100)
    num_elements_to_remove = int(num_blocks_to_remove * block_size)
    block_indices = np.random.choice(num_blocks, size=num_blocks_to_remove, replace=False)
    # slices_to_keep = [np.s_[:, :, i*block_size:(i+1)*block_size, :] for i in range(num_blocks_to_remove, num_blocks)]
    slices_to_keep = []
    for i in range(num_blocks):
        if i not in block_indices:
            slices_to_keep.append(np.s_[:, :, i*block_size:(i+1)*block_size, :])
    newData = np.concatenate([data[slice] for slice in slices_to_keep], axis=2)
    return newData

def makeFiles(data, labels, idxs, path, path2, batch_size, x_shuffle, y_shuffle, dataset, numOut, ds):
    inds = idxs
    ind_arrays = []
    labels_shuf = None
    if y_shuffle == True:
        labels_shuf = np.random.permutation(labels)
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
        if ds != None and ds >= 0 and ds <= 99:
            batch_data = downsampleData(batch_data, ds)
            np.savez(f"{path.rstrip('/')}/{idx}_{dataset}_data.npz", **{"X": batch_data, "y": batch_labs})
        if x_shuffle == True and path2 != None:
            indices = np.random.permutation(batch_data.shape[2])
            batch_data_shuf = batch_data[:,:,indices,:]
            np.savez(f"{path.rstrip('/')}/{idx}_{dataset}_data.npz", **{"X": batch_data, "y": batch_labs})
            np.savez(f"{path2.rstrip('/')}/{idx}_{dataset}_data.npz", **{"X": batch_data_shuf, "y": batch_labs})
        elif y_shuffle == True and path2 != None:
            batch_labs_shuf = labels_shuf[batch_idxs]
            np.savez(f"{path.rstrip('/')}/{idx}_{dataset}_data.npz", **{"X": batch_data, "y": batch_labs})
            np.savez(f"{path2.rstrip('/')}/{idx}_{dataset}_data.npz", **{"X": batch_data, "y": batch_labs_shuf})
        else:
            np.savez(f"{path.rstrip('/')}/{idx}_{dataset}_data.npz", **{"X": batch_data, "y": batch_labs})
        
def main():
    parser = argparse.ArgumentParser(description='Splits data to be passed as input to the data generator for Keras')
    parser.add_argument( '--config-file', '-cf', help = "Configuration filepath",dest='configfile', required = True, type=str)
    parser.add_argument( '--config', '-c', help = "Set of configurations",dest='config', required = True, type=str)
    parser.add_argument( '--downsample', '-ds', help = "Downsample (true/false)",dest='downsample', type=int, required = False)
    args = parser.parse_args()
    
    cfgFile = args.configfile
    cfg = args.config
    dataset = gu.readMultiConfig(cfgFile, cfg)

    dataPath = dataset['data']
    labelsPath = dataset['labels']
    batchsize = dataset['batch_size']
    numOutput = dataset['num_output_files']
    xshuffle = dataset['x_shuffle']
    yshuffle = dataset['y_shuffle']
    outputPath = dataset['output_dir']
    outputPath2 = dataset['output_dir_2']
    downsample = dataset.get('downsample')

    remove_files_if_directory_not_empty(outputPath)
    if outputPath2 != None:
        remove_files_if_directory_not_empty(outputPath2)
    
    print(f"Loading data: {dataPath.split('/')[-1]} and {labelsPath.split('/')[-1]} with options x_shuffle = {xshuffle}, y_shuffle = {yshuffle}")

    X = np.load(dataPath, mmap_mode="r")
    y = np.loadtxt(labelsPath)

    if yshuffle == True and outputPath2 != None:
        y = np.random.permutation(y)

    train_idxs, test_idxs, train_labs, test_labs = train_test_split(range(len(y)), y, test_size=(int(len(y) / 3)), stratify=y)

    makeFiles(X, y, test_idxs, outputPath, 
            outputPath2, batchsize, xshuffle, 
            yshuffle, 'test', numOutput, downsample)
    makeFiles(X, y, train_idxs, outputPath,
            outputPath2, batchsize, xshuffle,
            yshuffle, 'train', numOutput, downsample)

if __name__ == '__main__':
    main()
    