import copy
import json
import yaml
from sklearn.model_selection import ParameterGrid


def expand_config(base_file, hyper_file, output_file):
    """
    This is useful for taking the qanta.yaml config, a set of values to try for different hyper parameters, and
    generating a configuration representing each value in the parameter sweep
    """
    with open(base_file) as f:
        base_conf = yaml.load(f)

    with open(hyper_file) as f:
        hyper_conf = yaml.load(f)

    all_base_guessers = base_conf["guessers"]
    final_guessers = {}

    for guesser, params in hyper_conf["parameters"].items():
        base_guesser_conf = all_base_guessers[guesser]
        if len(base_guesser_conf) != 1:
            raise ValueError(
                "More than one configuration for parameter tuning base is invalid"
            )
        base_guesser_conf = base_guesser_conf[0]

        parameter_set = set(base_guesser_conf.keys()) | set(params.keys())
        param_grid = {}
        for p in parameter_set:
            if p in params:
                param_grid[p] = params[p]
            else:
                param_grid[p] = [base_guesser_conf[p]]

        parameter_list = list(ParameterGrid(param_grid))
        final_guessers[guesser] = parameter_list

    final_conf = copy.deepcopy(base_conf)
    for g in final_conf["guessers"]:
        if g in final_guessers:
            final_conf["guessers"][g] = copy.deepcopy(final_guessers[g])

    # There is a bug in yaml.dump that doesn't handle outputting nested dicts/arrays correctly. I didn't want to debug
    # So instead output to json then convert that to yaml
    with open("/tmp/qanta-tmp.json", "w") as f:
        json.dump(final_conf, f)

    with open("/tmp/qanta-tmp.json") as f:
        conf = json.load(f)

    with open(output_file, "w") as f:
        yaml.dump(conf, f)
