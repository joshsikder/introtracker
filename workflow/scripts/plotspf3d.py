import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotAll():
    a1 = np.load('/nas/longleaf/home/jsikder/introtracker/workflow/datasets/rfData/tree1.npz')
    a2 = np.load('/nas/longleaf/home/jsikder/introtracker/workflow/datasets/rfData/tree2.npz')
    a3 = np.load('/nas/longleaf/home/jsikder/introtracker/workflow/datasets/rfData/tree3.npz')
    a4 = np.load('/nas/longleaf/home/jsikder/introtracker/workflow/datasets/rfData/tree4.npz')
    
    a1 = a1['X']
    a2 = a2['X']
    a3 = a3['X']
    a4 = a4['X']

    data1 = a1[:, [1, 4, 5]]
    data2 = a2[:, [1, 4, 5]]
    data3 = a3[:, [1, 4, 5]]
    data4 = a4[:, [1, 4, 5]]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], color='r', alpha=0.5, label='Tree 1')
    ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], color='g', alpha=0.5, label='Tree 2')
    ax.scatter(data3[:, 0], data3[:, 1], data3[:, 2], color='b', alpha=0.5, label='Tree 3')
    ax.scatter(data4[:, 0], data4[:, 1], data4[:, 2], color='y', alpha=0.5, label='Tree 4')

    ax.set_title('All Trees')
    ax.set_xlabel('001')
    ax.set_ylabel('010')
    ax.set_zlabel('011')

    plt.legend()

    plt.savefig('/nas/longleaf/home/jsikder/introtracker/workflow/temp/3d_plot.png', dpi=300)
    pass

def plotTree_4(a, b, c, d, filename):
    a1 = np.load(f'/nas/longleaf/home/jsikder/introtracker/workflow/datasets/spfs/{a}_spf.npy')
    a2 = np.load(f'/nas/longleaf/home/jsikder/introtracker/workflow/datasets/spfs/{b}_spf.npy')
    a3 = np.load(f'/nas/longleaf/home/jsikder/introtracker/workflow/datasets/spfs/{c}_spf.npy')
    a4 = np.load(f'/nas/longleaf/home/jsikder/introtracker/workflow/datasets/spfs/{d}_spf.npy')
    
    data1 = a1[:, [1, 4, 5]]
    data2 = a2[:, [1, 4, 5]]
    data3 = a3[:, [1, 4, 5]]
    data4 = a4[:, [1, 4, 5]]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], color='r', alpha=0.5, label=f'{a}')
    ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], color='g', alpha=0.5, label=f'{b}')
    ax.scatter(data3[:, 0], data3[:, 1], data3[:, 2], color='b', alpha=0.5, label=f'{c}')
    ax.scatter(data4[:, 0], data4[:, 1], data4[:, 2], color='y', alpha=0.5, label=f'{d}')

    ax.set_title(f'{filename}')
    ax.set_xlabel('001')
    ax.set_ylabel('010')
    ax.set_zlabel('011')

    plt.legend()

    plt.savefig(f'/nas/longleaf/home/jsikder/introtracker/workflow/temp/3d_plot_{filename}.png', dpi=300)

def plotTree_3(a, b, c, filename):
    a1 = np.load(f'/nas/longleaf/home/jsikder/introtracker/workflow/datasets/spfs/{a}_spf.npy')
    a2 = np.load(f'/nas/longleaf/home/jsikder/introtracker/workflow/datasets/spfs/{b}_spf.npy')
    a3 = np.load(f'/nas/longleaf/home/jsikder/introtracker/workflow/datasets/spfs/{c}_spf.npy')
    
    data1 = a1[:, [1, 4, 5]]
    data2 = a2[:, [1, 4, 5]]
    data3 = a3[:, [1, 4, 5]]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], color='r', alpha=0.5, label=f'{a}')
    ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], color='g', alpha=0.5, label=f'{b}')
    ax.scatter(data3[:, 0], data3[:, 1], data3[:, 2], color='b', alpha=0.5, label=f'{c}')

    ax.set_title(f'{filename}')
    ax.set_xlabel('001')
    ax.set_ylabel('010')
    ax.set_zlabel('011')

    plt.legend()

    plt.savefig(f'/nas/longleaf/home/jsikder/introtracker/workflow/temp/3d_plot_{filename}.png', dpi=300)


def main():
    plotAll()
    plotTree_4('simulation0', 'simulation1', 'simulation2', 'simulation3', 'tree1')
    plotTree_3('simulation4', 'simulation5', 'simulation6', 'tree2')
    plotTree_3('simulation7', 'simulation8', 'simulation9', 'tree3')
    plotTree_4('simulation10', 'simulation11', 'simulation12', 'simulation13', 'tree4')

if __name__ == '__main__':
    main()
