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

    all_base_guessers = base_conf['guessers']
    final_guessers = {}

    for guesser, params in hyper_conf['parameters'].items():
        base_guesser_conf = all_base_guessers[guesser]
        if len(base_guesser_conf) != 1:
            raise ValueError('More than one configuration for parameter tuning base is invalid')
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

    for g in base_conf['guessers']:
        if g in final_guessers:
            base_conf['guessers'][g] = final_guessers[g]

    with open(output_file, 'w') as f:
        yaml.dump(base_conf, f)

