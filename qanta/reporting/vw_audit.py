import re
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sb
from qanta.reporting.report_generator import ReportGenerator
from qanta.util.constants import VW_AUDIT_REGRESSOR_REPORT
from qanta.util.io import safe_path


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


def audit_report(df, output):
    df.to_csv(output)

    df.head(25).plot.bar('feature', 'value')
    plt.title('Feature Magnitudes')
    plt.xlabel('Magnitude')
    plt.savefig('/tmp/feature_importance.png', dpi=200, format='png')

    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_colwidth', 30)

    top_features = str(df.head(100))

    report = ReportGenerator({
        'feature_importance_plot': '/tmp/feature_importance.png',
        'top_features': top_features
    }, 'audit_regressor.md')

    output = safe_path(VW_AUDIT_REGRESSOR_REPORT)
    report.create(output)
    plt.clf()
    plt.cla()
    plt.close()
