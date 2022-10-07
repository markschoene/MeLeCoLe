import yaml


class Config:
    def __init__(self, filename):
        assert type(filename) == str and (".yml" in filename or ".yaml" in filename), "Must be string ending with .yml or .yaml"

        self.filename = filename
        self.config = dict()

        if filename:
            self.load(self.filename)

    def load(self, filename=None):
        if not filename:
            filename = self.filename

        with open(filename, 'r') as f:
            cfg = yaml.safe_load(f)
            if len(self.config) == 0:
                self.config = cfg
            else:
                self.config.update(cfg)
