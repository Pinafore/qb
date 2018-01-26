import copy
from typing import Dict
from sklearn.model_selection import ParameterGrid


def generate_configs(original_conf: Dict, sweep_conf):
    """
    This is useful for taking the qanta.yaml config, a set of values to try for different hyper parameters, and
    generating a configuration representing each value in the parameter sweep
    """

    param_grid = {}

    for hp in sweep_conf['parameters']:
        hp_name = '.'.join(hp['access'])
        bound_values = []
        for v in hp['values']:
            bound_values.append((v, hp['access']))
        param_grid[hp_name] = bound_values

    parameter_list = list(ParameterGrid(param_grid))
    configurations = []
    for param_spec in parameter_list:
        new_conf = copy.deepcopy(original_conf)
        for param_value, param_access in param_spec.values():
            curr_obj = new_conf
            for key in param_access[:-1]:
                curr_obj = curr_obj[key]
            curr_obj[param_access[-1]] = param_value
        configurations.append(new_conf)

    return configurations
