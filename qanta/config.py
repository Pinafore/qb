import os
import yaml
from qanta.util.environment import data_path


DEFAULT_CONFIG = data_path("qanta-defaults.yaml")
CUSTOM_CONFIG = data_path("qanta.yaml")


def load_config():
    if os.path.exists(CUSTOM_CONFIG):
        with open(CUSTOM_CONFIG) as f:
            return yaml.load(f)
    elif os.path.exists(DEFAULT_CONFIG):
        with open(DEFAULT_CONFIG) as f:
            return yaml.load(f)
    elif os.path.exists("/ssd-c/qanta/qb/qanta.yaml"):
        with open("/ssd-c/qanta/qb/qanta.yaml") as f:
            return yaml.load(f)
    elif os.path.exists("/ssd-c/qanta/qb/qanta-defaults.yaml"):
        with open("/ssd-c/qanta/qb/qanta-defaults.yaml") as f:
            return yaml.load(f)
    else:
        raise ValueError(
            "Qanta YAML configuration could not be found in qanta-defaults.yaml or qanta.yaml"
        )


conf = load_config()
