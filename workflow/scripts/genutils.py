import yaml

class GeneralUtilities:
    def readConfig(configPath):
        with open(configPath, "r") as infile:
            yamldata = yaml.safe_load(infile)
        return yamldata

    def readMultiConfig(configPath):
        with open(configPath, "r") as infile:
            yamldata = yaml.safe_load_all(infile)
        return yamldata
