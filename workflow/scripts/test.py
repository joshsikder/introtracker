import os
import sys
import glob

#files = os.listdir(f"/nas/longleaf/home/jsikder/phylo1/workflow/data/simulations/{sys.argv[1]}/{sys.argv[2]}/{sys.argv[3]}")
files = glob.glob(f'../data/simulations/{sys.argv[1]}/{sys.argv[2]}/{sys.argv[3]}*')
print(files)