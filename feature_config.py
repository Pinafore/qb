from collections import defaultdict, OrderedDict

kFEATURES = OrderedDict([
    # ("ir", None),
    ("lm", None),
    ("deep", None),
    ("answer_present", None),
    ("text", None),
    # ("classifier", None),
    ("wikilinks", None),
    # ("mentions", None),
    ])

if __name__ == "__main__":
    import argparse
    # Generate a list of all of the relevant files given fold and granularity
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--fold', type=str, default='dev',
                        help='Data fold to give feature files for')
    parser.add_argument('--granularity', type=str, default='sentence',
                        help='Granularity to give feature files for')
    flags = parser.parse_args()

    print(" ".join("features/%s/%s.%s.feat" % (flags.fold, flags.granularity, x)
                   for x in kFEATURES))

