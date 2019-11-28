import yaml

def getConfig(file):
    with open("./configs/" + file + ".yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg
