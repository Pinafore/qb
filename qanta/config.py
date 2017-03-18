import os
import hcl


DEFAULT_CONFIG = 'qanta-defaults.hcl'
CUSTOM_CONFIG = 'qanta.hcl'


def load_config():
    if os.path.exists(CUSTOM_CONFIG):
        with open(CUSTOM_CONFIG) as f:
            loaded_conf = hcl.load(f)
    else:
        with open(DEFAULT_CONFIG) as f:
            loaded_conf = hcl.load(f)
    return loaded_conf


conf = load_config()
