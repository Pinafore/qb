from csv import DictWriter
import os
from glob import glob

from ingestion.create_db import NaqtQuestion
from ingestion.page_assigner import PageAssigner
from qanta.datasets.quiz_bowl import QuestionDatabase

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Import questions')
    parser.add_argument('--naqt_path', type=str, default='')
    parser.add_argument('--ambiguous_path', type=str,
                        default='data/internal/page_assignment/ambiguous/')
    parser.add_argument('--unambiguous_path', type=str,
                        default='data/internal/page_assignment/unambiguous/')
    parser.add_argument('--csv_out', default="2016_hsnct", type=str)
    parser.add_argument('--id_offset', default=7000000, type=int)
    flags = parser.parse_args()
    
    unmapped = set()

    from nltk.tokenize.treebank import TreebankWordTokenizer
    tk = TreebankWordTokenizer().tokenize
    
    pa = PageAssigner(QuestionDatabase.normalize_answer)
    for ii in glob("%s/*" % flags.ambiguous_path):
        pa.load_ambiguous(ii)
    for ii in glob("%s/*" % flags.unambiguous_path):
        pa.load_unambiguous(ii)
    
    question_id = flags.id_offset
    with open("%s.csv" % flags.csv_out, 'w') as outfile:
        writer = DictWriter(outfile, ["id", "answer", "text"])
        writer.writeheader()
        
        if flags.naqt_path:
            for qq in NaqtQuestion.naqt_reader(flags.naqt_path):
                if not qq.text:
                    log.info("Bad question %s" % str(qq.metadata["ID"]))
                    num_skipped += 1
                    continue

                
                page = pa(qq.answer, tk(qq.text))
                if page != "":
                    writer.writerow({"id": question_id,
                                     "answer": page,
                                     "text": qq.clean_naqt(qq.text)})
                    question_id += 1
                else:
                    norm = QuestionDatabase.normalize_answer(qq.answer)
                    unmapped.add(norm)

    print("%i questions written")
    print("Unmapped:")
    for ii in unmapped:
        print("%s\t" % ii)
    
