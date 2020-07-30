from typing import Set
import json
from qanta.datasets.abstract import AbstractDataset, TrainingData


class TriviaQADataset(AbstractDataset):
    def __init__(self, answers: Set[str]):
        super().__init__()
        self.answers = answers

    def training_data(self):
        with open("data/external/unfiltered-web-train.json") as f:
            train = json.load(f)["Data"]

        wiki_train = [q for q in train if q["Answer"]["Type"] == "WikipediaEntity"]
        x_train = [q["Question"] for q in wiki_train]
        y_train = [q["Answer"]["MatchedWikiEntityName"] for q in wiki_train]
        questions = []
        answers = []
        for q, a in zip(x_train, y_train):
            if a in self.answers:
                questions.append([q])
                answers.append(a)

        return questions, answers, None
