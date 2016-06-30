# File to take a label file and create multiple version with different weights
# for negative answers
import sys

NEG_WEIGHTS = [2., 4., 8., 16., 32., 64.]

if __name__ == "__main__":
    infilename = sys.argv[1]

    for weight in NEG_WEIGHTS:
        fname = infilename.replace(".feat", ".%s" % str(int(weight)))
        o = open(fname, 'w')
        print(fname)
        for feat_line in open(infilename):
            label = feat_line.split()[0]
            neg_count_str = feat_line.split()[1]

            if int(label) == 1:
                o.write(feat_line)
            else:
                neg_count = int(neg_count_str)
                new = feat_line.replace(" %s '" % neg_count_str,
                                        " %f '" % (weight / float(neg_count)))
                # print(neg_count, ww, ww / float(neg_count), new)
                o.write(new)
        o.close()
