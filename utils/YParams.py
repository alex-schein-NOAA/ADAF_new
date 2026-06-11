import argparse
from ruamel.yaml import YAML


class YParams:
    """Yaml file parser"""

    def __init__(self, yaml_filename, config_name="EncDec", print_params=False):
        self._yaml_filename = yaml_filename
        self._config_name = config_name #(2026-06-11) Currently only "EncDec" is supported, but leaving this functionality here in case it's ever useful
        self.params = {}

        with open(yaml_filename, "rb") as _file:
            yaml = YAML().load(_file)
            for key, val in yaml[config_name].items():
                if print_params:
                    print(key, val)
                if val == "None":
                    val = None

                self.params[key] = val
                self.__setattr__(key, val)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key):
        return key in self.params

    def update_params(self, config):
        for key, val in config.items():
            self.params[key] = val
            self.__setattr__(key, val)

    def override_from_cli(self, args):
        """
        Intercepts command-line arguments and overrides matching YAML parameters.
        Ignored if the CLI argument is None or doesn't exist in the YAML.
        """
        # Convert argparse Namespace to a standard dictionary if needed
        args_dict = vars(args) if isinstance(args, argparse.Namespace) else args
        
        for key, val in args_dict.items():
            # Only override if the key exists in YAML and the user provided a value
            if key in self.params and val is not None:
                self.params[key] = val
                self.__setattr__(key, val)

