import sys, errno


def format_audit(n_features):
    try:
        while True:
            score, qid = next(sys.stdin).split()
            features = next(sys.stdin).split()
            sys.stdout.write(qid + '\t' + ' '.join(features[0:n_features]) + '\n')
    except StopIteration:
        return
    except IOError as e:
        if e.errno == errno.EPIPE:
            return
