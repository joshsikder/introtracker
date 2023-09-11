import yaml
import argparse
class GeneralUtilities:
    def readConfig(configPath):
        with open(configPath, "r") as infile:
            yamldata = yaml.safe_load(infile)
        return yamldata

    def readMultiConfig(configPath, configName):
        with open(configPath, "r") as infile:
            yamldata = yaml.safe_load_all(infile)
            datasets = {}
            for set in yamldata:
                datasets[set[0]['name']] = {key: set[0][key] for key in list(set[0].keys())[1:]}
            return datasets[configName]

def main():
    parser = argparse.ArgumentParser(description='Keras model for detecting introgression')
    parser.add_argument( '--configfile', '-cf', help = "Configuration file",dest='cfgFile')
    parser.add_argument( '--config', '-c', help = "Configuration",dest='cfg')
    args = parser.parse_args()

    datasets = GeneralUtilities.readMultiConfig(args.cfgFile, args.cfg)

if __name__ == '__main__':
    main()