#!/usr/bin/env python3

from collections import defaultdict
import time
from unidecode import unidecode
from util.build_whoosh import text_iterator

from clm.lm_wrapper import LanguageModelWriter
from qanta.util.environment import QB_QUESTION_DB, data_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--wiki_location', type=str, default=data_path('data/wikipedia'))
    parser.add_argument('--question_db', type=str, default=QB_QUESTION_DB)
    parser.add_argument('--source_location', type=str, default=data_path('data/source'))
    parser.add_argument('--global_lms', type=int, default=5,
                        help="The number of background LMs we maintain")
    parser.add_argument('--vocab_size', type=int, default=100000)
    parser.add_argument("--min_answers", type=int, default=-1,
                        help="How many answers needed before including in LM")
    parser.add_argument("--max_pages", type=int, default=-1,
                        help="How many pages to add to the index")
    parser.add_argument("--stats_pages", type=int, default=5000,
                        help="How many pages to use for computing stats")
    parser.add_argument("--lm_out", type=str, default=data_path('data/lm.txt'))
    flags = parser.parse_args()

    min_answers = flags.min_answers
    print("Training language model with pages that appear more than %i times" %
          min_answers)

    lm = LanguageModelWriter(flags.vocab_size, flags.global_lms)
    num_docs = 0
    background = defaultdict(int)
    # Initialize language models
    for title, text in text_iterator(True, flags.wiki_location,
                                     True, flags.question_db,
                                     True, flags.source_location,
                                     flags.max_pages,
                                     min_pages=min_answers):
        num_docs += 1
        if num_docs % 500 == 0:
            print(unidecode(title), num_docs)
            print(list(lm.tokenize_without_censor(text[100:200])))

        for tt in lm.tokenize_without_censor(text):
            background[tt] += 1

    # Create the vocabulary
    for ii in background:
        lm.train_seen(ii, background[ii])
    vocab = lm.finalize()
    print(str(vocab)[:80])
    print("Vocab size is %i from %i docs" %
          (len(vocab), num_docs))
    del background

    # Train the language model
    doc_num = 0
    for corpus, qb, wiki, source in [("wiki", False, True, False),
                                     ("qb", True, False, False),
                                     ("source", False, False, True)
                                     ]:
        # Add training data
        start = time.time()
        for title, text in text_iterator(wiki, flags.wiki_location,
                                         qb, flags.question_db,
                                         source, flags.source_location,
                                         flags.max_pages,
                                         min_pages=min_answers):
            norm_title = lm.normalize_title(corpus, title)
            doc_num += 1
            if doc_num % 500 == 0 or time.time() - start > 10:
                print("Adding train doc %i, %s (%s)" %
                      (doc_num, unidecode(title), corpus))
                start = time.time()
            lm.add_train(norm_title, text)
            lm.add_train("compare_%i" % lm.compare(norm_title), text)

    print("Done training")
    if flags.lm_out:
        # Create the extractor object and write out the pickle
        o = open(flags.lm_out, 'w')
        lm.write_lm(o)
