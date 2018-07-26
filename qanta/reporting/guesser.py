import glob
import json
import pickle
import pandas as pd
from functional import pseq


def parse_report(path):
    with open(path, 'rb') as f:
        prp = pickle.load(f)
        config_num = int(path.split('/')[3])
        return {
            'first_accuracy': prp['first_accuracy'],
            'first_recall': prp['first_recall'],
            'full_accuracy': prp['full_accuracy'],
            'full_recall': prp['full_recall'],
            'guesser_name': prp['guesser_name'],
            'guesser_params': prp['guesser_params'],
            'config_num': config_num
        }


def read_guesser_reports(guesser):
    report_paths = glob.glob(f'output/guesser/{guesser}/*/guesser_report_guessdev.pickle', recursive=True)
    reports = pseq(report_paths).map(parse_report).list()
    hyper_params = set()
    fake_params = {'random_seed', 'training_time', 'config_num'}

    for r in reports:
        for p in r['guesser_params']:
            if p not in fake_params:
                hyper_params.add(p)
    hyper_params = list(hyper_params)
    return reports, hyper_params


def reports_to_df(reports):
    experiment_rows = []
    for r in reports:
        row = {}
        row['first_accuracy'] = r['first_accuracy']
        row['full_accuracy'] = r['full_accuracy']
        row['config_num'] = r['config_num']
        for k, v in r['guesser_params'].items():
            if isinstance(v, list):
                v = tuple(v)
            elif isinstance(v, dict):
                v = json.dumps(v, sort_keys=True)
            row[k] = v
        row['guesser'] = r['guesser_name']
        row['n_experiment'] = 1
        experiment_rows.append(row)
    return pd.DataFrame.from_records(experiment_rows).sort_values('first_accuracy', ascending=False).reset_index()


def aggregate_report_df(report_df, hyper_params):
    cols = hyper_params + ['first_accuracy', 'full_accuracy', 'n_experiment']
    agg_df = report_df[cols].fillna(-1).groupby(hyper_params).agg({
        'first_accuracy': {
            'first_accuracy_max': 'max',
            'first_accuracy_std': 'std',
            'first_accuracy_mean': 'mean',
            'first_accuracy_min': 'min'
        },
        'full_accuracy': {
            'full_accuracy_max': 'max',
            'full_accuracy_std': 'std',
            'full_accuracy_mean': 'mean',
            'full_accuracy_min': 'min'
        },
        'n_experiment': {'n_experiment': 'sum'}
    })
    agg_df.columns = agg_df.columns.droplevel()
    agg_df = agg_df.reset_index().sort_values('first_accuracy_max', ascending=False)
    return agg_df


def merge_reports(guessers):
    all_reports = {}
    all_hyper_params = {}
    all_dfs = {}
    all_agg_dfs = {}

    for g in guessers:
        reports, hyper_params = read_guesser_reports(g)
        df = reports_to_df(reports)
        agg_df = aggregate_report_df(df, hyper_params)
        all_reports[g] = reports
        all_hyper_params[g] = hyper_params
        all_dfs[g] = df
        all_agg_dfs[g] = agg_df

    return all_reports, all_hyper_params, all_dfs, all_agg_dfs


def find_best_guessers(all_report_dfs):
    best_guesser_configs = {}
    for g, df in all_report_dfs.items():
        best_guesser_configs[g] = df.iloc[0].config_num
    return best_guesser_configs
