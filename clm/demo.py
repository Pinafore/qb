from lm_wrapper import LanguageModelReader, LanguageModelBase, pretty_debug
from csv import DictReader
from collections import defaultdict

if __name__ == "__main__":
    lm = LanguageModelReader("data/language_model")
    lm.init()

    results = defaultdict(str)
    sort = defaultdict(int)
    for ii in DictReader(open("data/expo.csv")):
        for corpus in ["qb", "wiki", "source"]:
            ans = LanguageModelBase.normalize_title("", ii["answer"])
            feat = lm.feature_line(corpus, ii["answer"], ii["text"])
            sort[ans] += len(feat.split())
            results[ans] += "%s\n" % feat
            results[ans] += pretty_debug(corpus,
                                         lm.verbose_feature(corpus,
                                                            ii["answer"],
                                                            ii["text"]))
            results[ans] += "\n----------\n"

    for ii in sorted(results, key=lambda x: sort[x], reverse=True):
        print("==================")
        print(ii)
        print("==================")
        print(results[ii])
        print("==================")
