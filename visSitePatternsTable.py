import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import argparse
import os

def stackArrays(arr_dir):
    my_arrays_64 = []
    my_arrays_256 = []
    # files = [f"{arr_dir}simulation{i}_spf.npy" for i in range(1,14)]
    files = ["temp/testNewSitePatterns.npy"]
    for item in files:
        meanArr = np.mean(np.load(item), axis=0)
        if meanArr.shape[0] == 64:
            my_arrays_64.append(meanArr)
        elif meanArr.shape[0] == 256:
            my_arrays_256.append(meanArr)
    return(my_arrays_64, my_arrays_256, len(my_arrays_64), len(my_arrays_256))

def loadArray(arr):
    my_array = np.load(arr)
    indices = np.random.choice(my_array.shape[0], size=10, replace=False)
    sampled_array = my_array[indices]
    print(sampled_array.shape)
    return(sampled_array)

def createLabels(numTaxa):
    x_labels = list(product(['A','T','C','G'],repeat = numTaxa))
    x_labels = [''.join(i) for i in x_labels]
    return(x_labels)

def createFigure(array,labels,outPath, n):
    df = pd.DataFrame(array, index=['Simulation {}'.format(i) for i in range(1, n+1)], columns=labels)
    fig, ax = plt.subplots(figsize=(20, 10)) # adjust this as needed
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc = 'center', loc='center')
    plt.savefig(outPath, dpi=600)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='numeric2pattern conversion Ready for Keras')
    parser.add_argument( '--spf', help = "Directory containing npy arrays",dest='SPF')
    # parser.add_argument( '--arr1', help = "Directory containing npy arrays",dest='ARR1')
    # parser.add_argument( '--arr2', help = "Directory containing npy arrays",dest='ARR2')
    parser.add_argument( '--output', help = "Directory in which to save the output figure",dest='OUTPUT')
    args = parser.parse_args()

    arrays_64, arrays_256, num64, num256 = stackArrays(args.SPF)
    labels_64, labels_256 = createLabels(3), createLabels(4)
    createFigure(arrays_64, labels_64, os.path.join(args.OUTPUT, "spf_64_table.png"), num64)
    createFigure(arrays_256, labels_256, os.path.join(args.OUTPUT, "spf_256_table.png"), num256)
    # createFigure(loadArray(args.ARR1), labels_64, os.path.join(args.OUTPUT, "spf_64.png"),10)
    # createFigure(loadArray(args.ARR2), labels_256, os.path.join(args.OUTPUT, "spf_256.png"),10)

if __name__ == "__main__":
    main()