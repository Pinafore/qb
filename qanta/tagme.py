import pickle
from typing import List
import math

import luigi
import tagme
import progressbar
import multiprocessing

from qanta.util.environment import TAGME_GCUBE_TOKEN
from qanta.util.io import make_dirs
from qanta.datasets.quiz_bowl import QuizBowlDataset


BATCH_SIZE = 200


def run_tagme(instances: List[str], n_workers=None):
    tagme.GCUBE_TOKEN = TAGME_GCUBE_TOKEN
    if n_workers is None:
        pool = multiprocessing.Pool()
    else:
        pool = multiprocessing.Pool(n_workers)
    bar = progressbar.ProgressBar(max_value=len(instances))
    annotations = []
    for a in bar(pool.imap(tagme.annotate, instances, chunksize=2)):
        annotations.append(a)
    return annotations


def annotation_to_dict(response):
    return [
        {
            'begin': a.begin, 'end': a.end,
            'entity_id': a.entity_id, 'entity_title': a.entity_title,
            'mention': a.mention, 'score': a.score, 'uri': a.score
        }
        for a in response.annotations
    ]


class BatchQuestions(luigi.Task):
    def output(self):
        yield luigi.LocalTarget('output/tagme/batches.pickle')
        yield luigi.LocalTarget('output/tagme/meta.pickle')

    def run(self):
        make_dirs('output/tagme/')
        dataset = QuizBowlDataset(1, guesser_train=True)
        training_data = dataset.training_data()
        questions = training_data[0]
        batch = 0
        batch_lookup = {}

        while batch * BATCH_SIZE < len(questions):
            batch_questions = questions[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
            batch_lookup[batch] = batch_questions
            batch += 1

        with open('output/tagme/batches.pickle', 'wb') as f:
            pickle.dump(batch_lookup, f)

        with open('output/tagme/meta.pickle', 'wb') as f:
            pickle.dump(batch, f)


class TaggedQuestionBatch(luigi.Task):
    question_batch = luigi.IntParameter()

    def requires(self):
        yield BatchQuestions()

    def output(self):
        with open('output/tagme/meta.pickle', 'rb') as f:
            n_batches = pickle.load(f)

        for i in range(n_batches):
            yield luigi.LocalTarget('output/tagme/tagged_batch_{}.pickle'.format(self.question_batch))

    def run(self):
        tagme.GCUBE_TOKEN = TAGME_GCUBE_TOKEN
        with open('output/tagme/batches.pickle', 'rb') as f:
            batch_dict = pickle.load(f)
        batch_questions = batch_dict[self.question_batch]
        flattened_sentences = []
        for q in batch_questions:
            flattened_sentences.extend(q)

        annotation_responses = [tagme.annotate(s) for s in flattened_sentences]
        dict_annotations = [annotation_to_dict(ar) for ar in annotation_responses]
        with open('output/tagme/tagged_batch_{}.pickle'.format(self.question_batch), 'wb') as f:
            pickle.dump(dict_annotations, f)


class TaggedQuestions(luigi.WrapperTask):
    def requires(self):
        dataset = QuizBowlDataset(1, guesser_train=True)
        training_data = dataset.training_data()
        questions = training_data[0]
        n_batches = int(math.ceil(len(questions) / BATCH_SIZE))

        for i in range(n_batches):
            yield TaggedQuestionBatch(question_batch=i)
