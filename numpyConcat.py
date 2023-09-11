import numpy as np
import argparse

def concatArrays(arrays, output):
    np.save(output, np.concatenate(list(arrays.values())))

def main():
    parser = argparse.ArgumentParser(description='Numpy array concatenation script')
    parser.add_argument("input", nargs='+', help = "List of input arrays. Must be at least 2 arrays.")
    parser.add_argument( '--output', help = "Output file location",dest='OUTPUT',required=True)
    args = parser.parse_args()

    my_arrays = {}
    for i, item in enumerate(args.input):
        my_arrays[f"array{i+1}"] = np.load(item)
    
    concatArrays(my_arrays, args.OUTPUT)

if __name__ == "__main__":
    main()