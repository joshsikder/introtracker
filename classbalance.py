import numpy as np
from matplotlib import pyplot as plt
import os

tree1List, tree2List, tree3List, tree4List = [], [], [], []

basePath = '/nas/longleaf/home/jsikder/introtracker/workflow/datasets/'
for i in os.listdir(f'{basePath}tree1/'):
    a = np.load(f'{basePath}tree1/{i}')
    tree1List.append(int(a['y'][0]))

for i in os.listdir(f'{basePath}tree2/'):
    a = np.load(f'{basePath}tree2/{i}')
    tree2List.append(int(a['y'][0]))

for i in os.listdir(f'{basePath}tree3/'):
    a = np.load(f'{basePath}tree3/{i}')
    tree3List.append(int(a['y'][0]))

for i in os.listdir(f'{basePath}tree4/'):
    a = np.load(f'{basePath}tree4/{i}')
    tree4List.append(int(a['y'][0]))

hist1 = plt.hist(tree1List, bins=range(5), edgecolor='black', align='left')
hist2 = plt.hist(tree2List, bins=range(5), edgecolor='black', align='left')
hist3 = plt.hist(tree3List, bins=range(5), edgecolor='black', align='left')
hist4 = plt.hist(tree4List, bins=range(5), edgecolor='black', align='left')

hist1.xticks(range(4))
hist2.xticks(range(4))
hist3.xticks(range(4))
hist4.xticks(range(4))

hist1.title("Class balance: Tree 1")
hist2.title("Class balance: Tree 2")
hist3.title("Class balance: Tree 3")
hist4.title("Class balance: Tree 4")

hist1.xlabel("Class")
hist2.xlabel("Class")
hist3.xlabel("Class")
hist4.xlabel("Class")

hist1.ylabel("Frequency")
hist2.ylabel("Frequency")
hist3.ylabel("Frequency")
hist4.ylabel("Frequency")

hist1.savefig('/nas/longleaf/home/jsikder/introtracker/workflow/temp/classBalanceTree1.pdf')
hist2.savefig('/nas/longleaf/home/jsikder/introtracker/workflow/temp/classBalanceTree2.pdf')
hist3.savefig('/nas/longleaf/home/jsikder/introtracker/workflow/temp/classBalanceTree3.pdf')
hist4.savefig('/nas/longleaf/home/jsikder/introtracker/workflow/temp/classBalanceTree4.pdf')
