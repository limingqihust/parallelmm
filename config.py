import json

class Config:
    def __init__(self, path) -> None:
        with open(path, 'r') as infile:
            cfg = json.load(infile)
        
        self.cfg = cfg