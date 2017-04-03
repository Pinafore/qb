from typing import List, Tuple, Optional
import pickle
import os
import random
import re
from qanta.datasets.abstract import Answer, TrainingData, QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.preprocess import format_guess
from qanta.util.io import shell


def format_question(text):
    return re.sub(r'[^a-z0-9 ]+', '', text.lower())


class VWGuesser(AbstractGuesser):
    def __init__(self):
        super().__init__()
        self.label_to_i = None
        self.i_to_label = None
        self.max_label = None

    def qb_dataset(self):
        return QuizBowlDataset(2)

    def save(self, directory: str) -> None:
        model_path = os.path.join(directory, 'vw_guesser.model')
        shell('cp /tmp/vw_guesser.model {}'.format(model_path))
        data = {
            'label_to_i': self.label_to_i,
            'i_to_label': self.i_to_label,
            'max_label': self.max_label
        }
        data_pickle_path = os.path.join(directory, 'vw_guesser.pickle')
        with open(data_pickle_path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def targets(cls) -> List[str]:
        return ['vw_guesser.model', 'vw_guesser.pickle']

    def guess(self,
              questions: List[QuestionText],
              max_n_guesses: Optional[int]) -> List[List[Tuple[Answer, float]]]:
        #with open('/tmp/vw_test.txt', 'w') as f:
        #    for q in questions:
        #        features = format_question(q)
        #        f.write('1 |words {features}\n'.format(features=features))
        #shell('vw -t -i /tmp/vw_guesser.model -r /tmp/raw_predictions.txt -d /tmp/vw_test.txt')
        predictions = []
        with open('/tmp/raw_predictions.txt') as f:
            for line in f:
                all_label_scores = []
                for label_score in line.split():
                    label, score = label_score.split(':')
                    label = int(label)
                    score = float(score)
                    all_label_scores.append((self.i_to_label[label], score))
                top_label_scores = sorted(
                    all_label_scores, reverse=True, key=lambda x: x[1])[:max_n_guesses]
                predictions.append(top_label_scores)
        return predictions

    @classmethod
    def load(cls, directory: str):
        model_path = os.path.join(directory, 'vw_guesser.model')
        shell('cp {} /tmp/vw_guesser.model'.format(model_path))
        data_pickle_path = os.path.join(directory, 'vw_guesser.pickle')
        with open(data_pickle_path, 'rb') as f:
            data = pickle.load(f)
        guesser = VWGuesser()
        guesser.label_to_i = data['label_to_i']
        guesser.i_to_label = data['i_to_label']
        guesser.max_label = data['max_label']
        return guesser

    def train(self, training_data: TrainingData) -> None:
        questions = training_data[0]
        answers = [format_guess(g) for g in training_data[1]]

        x_data = []
        y_data = []
        for q, ans in zip(questions, answers):
            for sent in q:
                x_data.append(sent)
                y_data.append(ans)

        label_set = set(answers)
        self.label_to_i = {label: i for i, label in enumerate(label_set, 1)}
        self.i_to_label = {i: label for label, i in self.label_to_i.items()}
        self.max_label = len(self.label_to_i)

        with open('/tmp/vw_train.txt', 'w') as f:
            zipped = list(zip(x_data, y_data))
            random.shuffle(zipped)
            for x, y in zipped:
                features = format_question(x)
                label = self.label_to_i[y]
                f.write('{label} |words {features}\n'.format(label=label, features=features))

        shell('vw --oaa {max_label} -d /tmp/vw_train.txt -f /tmp/vw_guesser.model --loss_function '
              'logistic --ngram 2 --skips 1 -c --passes 10 -b 29'.format(max_label=self.max_label))
