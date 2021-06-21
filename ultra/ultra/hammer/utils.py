import yaml 

def read_config(file, head=None):
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    if head:
        return config[head]
    return config
