from collections import defaultdict, Counter
from math import log
from csv import DictWriter
import argparse
import operator

from numpy import percentile

kMAX_VAL = 1000


def lg(x):
    return log(x) / log(2.)


def entropy(x):
    if x == 0.0 or x == 1.0:
        return float("-inf")
    else:
        return -(x * lg(x) + (1. - x) * lg(1. - x))


class PurgingCounter(Counter):
    def __init__(self, max_size=1000000):
        self._max = max_size

    def __setitem__(self, item, val):
        if len(self) > self._max * 2:
            print("Purging")
            to_delete = self.most_common()[self._max:]
            for ii in to_delete:
                del self[ii]

        Counter.__setitem__(self, item, val)


class VwDiscreteReader:
    def __init__(self, input_file, name, label, num_vals=100, max_sent=5,
                 min_freq=3):
        self._observations = defaultdict(PurgingCounter)
        self._input_file = input_file
        self._label_file = label
        self._name = name
        self._max_sent = max_sent
        self._num_vals = num_vals
        self._min_freq = min_freq

        self._fields = ["feature", "name", "frequency", "sent", "ig"]

    def read_file(self):
        self._num_true = defaultdict(int)
        self._num_false = defaultdict(int)
        for feat, label in zip(open(self._input_file),
                               open(self._label_file)):
            fields = label.split()
            label = True if int(fields[0]) > 0 else False

            sent = int(float([x.split(":")[1] for x in fields
                              if x.startswith("sent:")][0]))

            if label:
                self._num_true[sent] += 1
            else:
                self._num_false[sent] += 1

            for jj in [x for x in feat.split()
                       if not ":" in x and not "|" in x]:
                self._observations[(label, sent)][jj] += 1

    def __iter__(self):
        scores = defaultdict(dict)

        base_entropy = {}
        total = {}
        for ss in self._num_true:
            base_entropy[ss] = entropy(self._num_true[ss] /
                                   float(self._num_false[ss]
                                         + self._num_true[ss]))
            total[ss] = self._num_true[ss] + self._num_false[ss]

        keys = self._observations.keys()
        # print("---------------------------")
        for label, sent in keys:
            for feat in self._observations[(label, sent)]:
                feature_total = self._observations[(True, sent)][feat] + \
                    self._observations[(False, sent)][feat]

                if feature_total == 0 or sent > self._max_sent:
                    continue

                present_prob = self._observations[(True, sent)][feat] \
                    / float(feature_total)
                present_entropy = entropy(present_prob)

                absent_prob = (self._num_true[sent] -
                               self._observations[(True, sent)][feat]) \
                               / float(total[sent] - feature_total)
                absent_entropy = entropy(absent_prob)

                # if present_prob > 0:
                #    print(label, sent, feat, present_prob, absent_prob,
                #          present_entropy, absent_entropy)

                if present_entropy < 0.0 and absent_entropy < 0.0:
                    continue

                ig = base_entropy[sent] \
                    - feature_total / float(total[sent]) * present_entropy \
                    - (total[sent] - feature_total) / float(total[sent])  * \
                    absent_entropy

                if feat in scores[sent]:
                    continue
                else:
                    scores[sent][feat] = min(kMAX_VAL, ig)

        # print("---------------------------")

        # high values
        for sent in scores:
            count = 0
            for feat, val in sorted(scores[sent].items(),
                                    key=operator.itemgetter(1)):
                freq = self._observations[(True, sent)][feat] + \
                    self._observations[(False, sent)][feat]

                if freq < self._min_freq:
                    continue

                d = {}
                d["frequency"] = freq
                d["feature"] = self._name
                d["name"] = feat
                d["ig"] = val
                d["sent"] = sent

                count += 1
                yield d

                if count > self._num_vals:
                    break


        # low value


class VwContReader:
    def __init__(self, input_file, name, label, sent_max=5, stat_buffer=10000,
                 low_percentile=0.0001, high_percentile=99.999):
        self._discrete = defaultdict(int)
        self._continuous = defaultdict(set)
        self._name = name
        self._input_file = input_file
        self._label_file = label
        self._sent_limit = sent_max
        self._stat_buffer = stat_buffer
        self._low_percentile = low_percentile
        self._high_percentile = high_percentile

        self._fields = ["correct", "sent", "guess", "value", "feature", "name"]

    def __iter__(self):

        # First read in some examples to get a range of values to print
        examples_read = 0
        stats_buffer = defaultdict(list)
        for feat, label in zip(open(self._input_file),
                               open(self._label_file)):
            examples_read += 1
            for ii in [x for x in feat.split() if ":" in x]:
                feature, val = ii.split(":")
                val = float(val)
                stats_buffer[feature].append(val)

            if examples_read > self._stat_buffer and \
                    all(len(stats_buffer[x]) > self._stat_buffer
                        for x in stats_buffer):
                break

        print(list(list(set(stats_buffer[x]))[:10] for x in stats_buffer))

        highs = {}
        lows = {}
        for ii in stats_buffer:
            lows[ii] = percentile(stats_buffer[ii], self._low_percentile)
            highs[ii] = percentile(stats_buffer[ii], self._high_percentile)

        print(lows)
        print(highs)

        # Now actually output
        for feat, label in zip(open(self._input_file),
                               open(self._label_file)):
            fields = label.split()
            d = {}
            d["correct"] = "correct" if int(fields[0]) > 0 else "wrong"
            d["sent"] = float([x.split(":")[1] for x in fields
                               if x.startswith("sent:")][0])

            if d["sent"] > self._sent_limit:
                continue

            guess_index = fields.index("|guess") + 1
            d["guess"] = fields[guess_index]
            d["feature"] = self._name

            for ii in [x for x in feat.split() if ":" in x]:
                feature, val = ii.split(":")
                val = float(val)
                d["name"] = feature
                d["value"] = val

                if val > lows[feature] and val < highs[feature]:
                    yield d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--feats', nargs='*', default=[],
                        help='')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument("--output_cont", type=str,
                        default="results/features_cont.csv",
                        help="Where we write output file")
    parser.add_argument("--output_disc", type=str,
                        default="results/features_disc.csv",
                        help="Where we write output file")


    flags = parser.parse_args()

    discrete = {}
    continuous = {}

    for feat in flags.feats:
        file = feat
        feat = feat.replace(".feat", "").rsplit(".", 1)[-1]
        print(file, feat)

        discrete[feat] = VwDiscreteReader(file, feat, flags.label)
        discrete[feat].read_file()

        continuous[feat] = VwContReader(file, feat, flags.label)

    o = DictWriter(open(flags.output_disc, 'w'),
                   fieldnames=discrete.values()[0]._fields)
    o.writeheader()

    for ii in discrete:
        for jj in discrete[ii]:
            o.writerow(jj)

    o = DictWriter(open(flags.output_cont, 'w'),
                   fieldnames=continuous.values()[0]._fields)
    o.writeheader()

    for ii in continuous:
        for jj in continuous[ii]:
            o.writerow(jj)
