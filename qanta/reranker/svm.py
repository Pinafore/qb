
from fuzzywuzzy import fuzz
from qanta.datasets.quiz_bowl import QuestionDatabase

class ExampleGenerator:
    """
    Generate the X value for our training data
    """
    
    def __init__(self):
        self._feat_index = 0
        self._feat_names = {}
        self._feat = []

    def add_feature(self, global_name, feat):
        self._feat.append(feat)
        for ii in feat.feature_names:
            self._feat_index += 1
            self._feat_names["%s:%s" % (global_name, ii)] = self._feat_index

    def __call__(self, question, guess):
        v = dlib.sparse_vector()
        for feat_type in self._feat:
            extracted = ff(question, guess)
            for feat_val in extracted:
                v.append(dlib.pair(self._feat_names[feat_val],
                                   extracted[feat_val]))
        return v

def create_train(example_generator, questions, guesses):
    data = dlib.ranking_pair()
    
    for gg in guesses.iterrows():
        guess = gg[1]["guess"]
        answer = questions[gg[1]["qnum"]].page
        text = questions[gg[1]["qnum"]].get_text(gg[1]["sent"],
                                                 gg[1]["token"])

        if guess == answer:
            data.relevant.append(example_generator(text, gg[1]))
        else:
            data.nonrelevant.append(example_generator(text, gg[1]))
    return data

def train_svm(data, c_val=10):
    trainer = dlib.svm_rank_trainer()
    trainer.c = c_val
    ranker = trainer.train(data)

    return ranker

class Feature:
    def __init__(self):
        self.feature_names = [""]
    
    def __len__(self):
        return 

class AnswerPresent(Feature):
    def __call__(self, text, title):
        Feature.__init__(self)
        d = {}
        if "(" in title:
            title = title[:title.find("(")].strip()
        val = fuzz.partial_ratio(title, text)
        d["raw"] = log(val + 1)
        d["length"] = log(val * len(title) / 100. + 1)

        longest_match = 1
        for ii in title.split():
            if ii.lower() in ENGLISH_STOP_WORDS:
                continue
            longest_match = max(longest_match, len(ii) if ii in text else 0)
        d["longest"] = log(longest_match)

        return d

    def feat_names(self):
        return ["raw", "length", "longest"]        

class RegexpFeature(Feature):
    def __init__(self, question_pattern, guess_pattern):
        Feature.__init__(self)        
        self._q = question_pattern
        self._a = guess_pattern

    def __call__(self, question, guess):
        None
        
class Reranker:
    def __init__(self):
        None

    def train(self, guesses):
        None

    def predict(self, guesses):
        None

if __name__ == "__main__":
    import pickle
    import argparse
    
    parser = argparse.ArgumentParser(description='Learn reranker')

    parser.add_argument('--db', type=str,
                        default='data/internal/2017_05_23.db')
    parser.add_argument('--train_fold', default="guessdev", type=str)
    parser.add_argument('--eval_fold', default="buzzerdev", type=str)
    
    flags = parser.parse_args()
    
    guesses = pickle.load(open("output/guesser/qanta.guesser.elasticsearch.ElasticSearchGuesser/guesses_%s.pickle" % flags.train_fold, 'rb'))
    qdb = QuestionDatabase(flags.db, load_expo=False)
    questions = qdb.all_questions()
