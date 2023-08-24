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
    def readKerasConfig(configPath):
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