
import argparse
from collections import defaultdict
from csv import DictWriter, DictReader

kSUM_FIELDS = ["sent", "weight", "vw", "token",
               "val", "type"]


class ResultCombiner:
    def __init__(self):
        self._total = 0
        self._right = defaultdict(int)
        self._wrong = defaultdict(int)
        self._hopeless = defaultdict(int)
        self._premature = defaultdict(int)
        self._late = defaultdict(int)

    def write(self, vw, weight):
        positions = set(self._right) | set(self._wrong) | set(self._hopeless)
        print("*", positions)
        d = {}
        d["weight"] = weight
        d["vw"] = vw
        total_right = 0
        total_wrong = 0
        total_hopeless = 0
        total_premature = 0
        total_late = 0
        for ss, tt in sorted(positions):
            total_right += self._right[(ss, tt)]
            total_wrong += self._wrong[(ss, tt)]
            total_hopeless += self._hopeless[(ss, tt)]
            total_premature += self._premature[(ss, tt)]
            total_late += self._late[(ss, tt)]
            d["sent"] = ss
            d["token"] = tt

            d["val"] = float(total_right) / float(self._total)
            d["type"] = "correct"
            yield d

            d["val"] = float(total_wrong) / float(self._total)
            d["type"] = "wrong"
            yield d

            d["val"] = float(total_hopeless) / float(self._total)
            d["type"] = "hopeless"
            yield d

            d["val"] = float(total_premature) / float(self._total)
            d["type"] = "premature"
            yield d

            d["val"] = float(total_late) / float(self._total)
            d["type"] = "late"
            yield d

            d["val"] = 1 - float(total_right + total_wrong +
                                 total_late + total_premature +
                                 total_hopeless) \
                / float(self._total)
            d["type"] = "unanswered"
            yield d

    def add(self, line):
        pos = (int(line["sentence"]), int(line["token"]))
        print(pos, line)
        self._total += 1
        pf = int(line["present_forward"])
        pb = int(line["present_backward"])
        if line["corr"] == "True":
            print("right")
            self._right[pos] += 1
        elif pf > 0:
            print("pre")
            self._premature[pos] += 1
        elif pb > 0:
            print("late")
            self._late[pos] += 1
        elif pf < 0 and pb < 0:
            print("hopeless")
            self._hopeless[pos] += 1
        else:
            print("wrong")
            self._wrong[pos] += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--perf', nargs='+', type=str)
    parser.add_argument('--output', type=str)

    flags = parser.parse_args()
    d = defaultdict(ResultCombiner)

    for ii in flags.perf:
        print(ii)
        fields = ii.split(".")
        vw = fields[-2]
        weight = fields[-3]
        for jj in DictReader(open(ii)):
            d[(vw, weight)].add(jj)

    o = DictWriter(open(flags.output, 'w'), kSUM_FIELDS)
    o.writeheader()
    for vw, weight in d:
        print(("FOLD", vw, weight))
        for jj in d[(vw, weight)].write(vw, weight):
            print(jj)
            o.writerow(jj)
