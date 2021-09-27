import yaml
import json

class Config:
    def __init__(self, config_fp=""):
        self.config_dict = {}
        if config_fp.endswith(".yaml"):
            with open(config_fp) as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
        elif config_fp.endswith(".json"):
            with open(config_fp) as f:
                self.config = json.load(f)
