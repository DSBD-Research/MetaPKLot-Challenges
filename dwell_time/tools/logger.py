import json

class logger():
    def __init__(self, folder, output_name):
        self.output_name = output_name
        self.folder = folder
        self.log = {}
    

    def add(self, keys, value):
        if not isinstance(keys, list):
            keys = [keys]
        
        current_level = self.log
        for key in keys[:-1]:
            if key not in current_level or not isinstance(current_level[key], dict):
                current_level[key] = {}
            current_level = current_level[key]

        current_level[keys[-1]] = value


    def save(self):
        with open(f"{self.folder}/{self.output_name}.json", "w") as output:
            json.dump(self.log, output, indent=2)