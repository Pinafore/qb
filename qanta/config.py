import os
import hcl
from qanta.util.environment import data_path


DEFAULT_CONFIG = data_path('qanta-defaults.hcl')
CUSTOM_CONFIG = data_path('qanta.hcl')


def load_config():
    if os.path.exists(CUSTOM_CONFIG):
        with open(CUSTOM_CONFIG) as f:
            return hcl.load(f)
    elif os.path.exists(DEFAULT_CONFIG):
        with open(DEFAULT_CONFIG) as f:
            return hcl.load(f)
    elif os.path.exists('/ssd-c/qanta/qb/qanta.hcl'):
        with open('/ssd-c/qanta/qb/qanta.hcl') as f:
            return hcl.load(f)
    elif os.path.exists('/ssd-c/qanta/qb/qanta-defaults.hcl'):
        with open('/ssd-c/qanta/qb/qanta-defaults.hcl') as f:
            return hcl.load(f)
    else:
        raise ValueError(
            'Qanta HCL configuration could not be found in qanta-defaults.hcl or qanta.hcl')


conf = load_config()
