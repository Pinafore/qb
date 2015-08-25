from collections import defaultdict
from csv import DictWriter
import argparse


class VwReader:
    def __init__(self, input_file, name, label):
        self._discrete = defaultdict(int)
        self._continuous = defaultdict(set)
        self._name = name
        self._input_file = input_file
        self._label_file = label

        self._fields = ["correct", "sent", "guess", "value", "feature", "name"]

    def __iter__(self):
        for feat, label in zip(open(self._input_file),
                               open(self._label_file)):
            fields = label.split()
            d = {}
            d["correct"] = int(fields[0])
            d["sent"] = float([x.split(":")[1] for x in fields
                               if x.startswith("sent:")][0])

            guess_index = fields.index("|guess") + 1
            d["guess"] = fields[guess_index]
            d["feature"] = self._name

            for ii in [x for x in feat.split() if ":" in x]:
                feature, val = ii.split(":")
                val = float(val)
                d["name"] = feature
                d["value"] = val

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

        readers[feat] = VwReader(file, feat, flags.label)

    o = DictWriter(open(flags.output, 'w'),
                   fieldnames=readers.values()[0]._fields)
    o.writeheader()

    print(readers)
    for ii in readers:
        for jj in readers[ii]:
            o.writerow(jj)
