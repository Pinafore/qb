import sys
import errno


def format_feature(feature_str):
    name, _, value, weight = feature_str.split(':')
    value = float(value)
    weight = float(weight)
    return name, value * weight


def format_audit(n_features):
    try:
        while True:
            score, qid = next(sys.stdin).split()
            features = sorted(
                [format_feature(f) for f in next(sys.stdin).split()],
                reverse=True,
                key=lambda f: abs(f[1])
            )[:n_features]
            feature_str = ' '.join(['{}:{:.2f}'.format(name, score) for name, score in features])
            sys.stdout.write(qid + '\t' + feature_str + '\n')
    except StopIteration:
        return
    except IOError as e:
        if e.errno == errno.EPIPE:
            return
