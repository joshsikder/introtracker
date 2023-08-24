import yaml

class GeneralUtilities:
    def readConfig(configPath):
        with open(configPath, "r") as infile:
            yamldata = yaml.safe_load(infile)
        return yamldata

    def readDatasetsConfig(configPath):
        with open(configPath, "r") as infile:
            yamldata = yaml.safe_load_all(infile)
            datasets = {}
            for set in yamldata:
                a = set[0]
                datasets[a['name']] = {
                    'data':a['data'], 
                    'labels':a['labels'], 
                    'batch_size':a['batch_size'], 
                    'num_output_files':a['num_output_files'],
                    'shuffle':a['shuffle'],
                    'output_dir':a['output_dir'],
                    'output_dir_2':a['output_dir_2'],
                }
            return datasets
                
#   name: temptree
#   data: datasets/treeMatrices/tree1_matrices.npy
#   labels: datasets/treeLabels/temptree.txt
#   batch_size: 1
#   num_output_files: 3000
#   shuffle: False
#   output_dir: /work/users/j/s/jsikder/fordan/temp3
        # return yamldata

def main():
    configPath = '/nas/longleaf/home/jsikder/introtracker/workflow/conf/dataSplitTemp.yaml'
    yamldata = GeneralUtilities.readMultiConfig(configPath)
    datasets = {}

    for i in yamldata:
        # print(type(i[0]))
        print(i[0])
        # datasets[i[0]] = i[1:]

if __name__ == '__main__':
    main()