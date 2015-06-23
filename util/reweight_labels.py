# File to take a label file and create multiple version with different weights
# for negative answers
import sys

kNEG_WEIGHTS = [2., 4., 8., 16., 32., 64.]

if __name__ == "__main__":
    infilename = sys.argv[1]

    for ww in kNEG_WEIGHTS:
        fname = infilename.replace(".feat", ".%s" % str(int(ww)))
        o = open(fname, 'w')
        print(fname)
        for ii in open(infilename):
            label = ii.split()[0]
            neg_count_str = ii.split()[1]

            if int(label) == 1:
                o.write(ii)
            else:
                neg_count = int(neg_count_str)
                new = ii.replace(" %s '" % neg_count_str,
                                 " %f '" % (ww / float(neg_count)))
                # print(neg_count, ww, ww / float(neg_count), new)
                o.write(new)
        o.close()
