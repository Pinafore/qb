from collections import defaultdict, Counter
from csv import DictWriter
from numpy import percentile
import argparse


class PurgingCounter(Counter):
    def __init__(self, max_size=1000000):
        self._max = max_size

    def __setitem__(self, item, val):
        if len(pg) > self._max * 2:
            print("Purging")
            to_delete = self.most_common()[self._max:]
            for ii in to_delete:
                del self[ii]

        Counter.__setitem__(self, item, val)


class VwDiscreteReader:
    def __init__(self, input_file, name, label):
        self._observations = defaultdict(PurgingCounter)
        self._input_file = input_file
        self._label_file = label
        self._name = name

        self._fields = ["feature", "name", "correct", "wrong"]

    def read_file(self):
        for feat, label in zip(open(self._input_file),
                               open(self._label_file)):
            fields = label.split()
            label = True if int(fields[0]) > 0 else "wrong"
            sent = int(float([x.split(":")[1] for x in fields
                              if x.startswith("sent:")][0]))

            for jj in [x for x in feat.split()
                       if not ":" in x and not "|" in x]:
                self._observations[(label, sent)] += 1


class VwContReader:
    def __init__(self, input_file, name, label, sent_max=5, stat_buffer=10000,
                 low_percentile=10, high_percentile=90):
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
            stats_buffer[feat].append(val)

            if examples_read > self._stat_buffer and \
                    all(len(stats_buffer[x]) > self._stat_buffer):
                break

        highs = {}
        lows = {}
        for ii in stats_buffer:
            lows[ii] = percentile(stats_buffer[ii], self._low_percentile)
            highs[ii] = percentile(stats_buffer[ii], self._high_percentile)

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
    parser.add_argument("--output", type=str,
                        default="results/features.csv",
                        help="Where we write output file")

    flags = parser.parse_args()

    readers = {}

    for feat in flags.feats:
        file = feat
        feat = feat.replace(".feat", "").rsplit(".", 1)[-1]
        print(file, feat)

        readers[feat] = VwContReader(file, feat, flags.label)

    o = DictWriter(open(flags.output, 'w'),
                   fieldnames=readers.values()[0]._fields)
    o.writeheader()

    print(readers)
    for ii in readers:
        for jj in readers[ii]:
            o.writerow(jj)
