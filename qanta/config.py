import os

import yaml

from qanta.util.environment import data_path

DEFAULT_CONFIG = data_path("qanta-defaults.yaml")
CUSTOM_CONFIG = data_path("qanta.yaml")


def load_config():
    if os.path.exists(CUSTOM_CONFIG):
        with open(CUSTOM_CONFIG) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    elif os.path.exists(DEFAULT_CONFIG):
        with open(DEFAULT_CONFIG) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise ValueError(
            "Qanta YAML configuration could not be found in qanta-defaults.yaml or qanta.yaml"
        )


conf = load_config()
