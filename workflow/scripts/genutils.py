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
                    'x_shuffle':a['x_shuffle'],
                    'y_shuffle':a['y_shuffle'],
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
                datasets[a['name']] = {'inputs':a['inputs'], 
                                       'output':a['output'], 
                                       'ntaxa':a['ntaxa'], 
                                       'loci_count':a['loci_count'],
                                       'loci_length':a['loci_length'],
                                       'n_classes':a['n_classes'],
                                       'n_channels':a['n_channels'],
                                       'shuffle':a['shuffle'],
                                       'labelCorrection':a['labelCorrection'],
                                       'batch_size':a['batch_size'],
                                       'epochs':a['epochs'],
                                       'train_start':a['train_start'],
                                       'train_end':a['train_end'],
                                       'val_start':a['val_start'],
                                       'val_end':a['val_end'],
                }
            return datasets