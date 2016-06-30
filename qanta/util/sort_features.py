import operator
import sys
from heapq import nlargest, nsmallest
from itertools import chain


def vw_tuple_iterator(filename):
    in_features = False
    for ii in open(filename):
        if ii.startswith("Constant"):
            in_features = True
        if not in_features:
            continue
        feat, key, value = ii.split(":")
        value = float(value)
        yield (value, feat)


def main():
    top = nlargest(1000, vw_tuple_iterator(sys.argv[1]))
    bottom = nsmallest(1000, vw_tuple_iterator(sys.argv[1]))

    features = {}
    for val, feat in chain(top, bottom):
        features[feat] = val

    # Open output file
    o = open(sys.argv[2], 'w')
    for feat, val in sorted(features.items(), key=operator.itemgetter(1)):
        o.write("%f\t%s\n" % (val, feat))

if __name__ == "__main__":
    main()
