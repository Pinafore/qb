import re
import pandas as pd


def parse_audit(path):
    columns = {'namespace': [], 'feature': [], 'f_id': [], 'value': [], 'magnitude': []}
    with open(path) as f:
        for line in f:
            names = re.findall('(?:[a-zA-Z\_]+\^)?[a-zA-Z\_\(\)]+', line)
            namespace_list = []
            feature_list = []
            for m in names:
                if '^' in m:
                    n, f = m.split('^')
                    namespace_list.append(n)
                    feature_list.append(f)
                else:
                    namespace_list.append('vw')
                    feature_list.append(m)

            namespace = '*'.join(namespace_list)
            feature = '*'.join(feature_list)

            _, f_id, value = line.split(':')
            f_id = int(f_id)
            value = float(value)
            columns['namespace'].append(namespace)
            columns['feature'].append(feature)
            columns['f_id'].append(f_id)
            columns['value'].append(value)
            columns['magnitude'].append(abs(value))

    return pd.DataFrame(columns).sort_values('magnitude', ascending=False)
