import pickle
import torch
from tqdm import tqdm
import prettytable

from drqa.reader import Predictor
import drqa.tokenizers

from qanta.bonus.data import BonusPairsDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser.experimental.elasticsearch_instance_of import (
    ElasticSearchWikidataGuesser,
)

drqa.tokenizers.set_default("corenlp_classpath", "corenlp/*")


def test():
    gspec = AbstractGuesser.list_enabled_guessers()[0]
    guesser_dir = AbstractGuesser.output_path(
        gspec.guesser_module, gspec.guesser_class, ""
    )
    guesser = ElasticSearchWikidataGuesser.load(guesser_dir)

    torch.cuda.set_device(0)
    predictor = Predictor()
    predictor.cuda()

    dataset = BonusPairsDataset()
    examples = [x for x in dataset.examples if x["start"] != -1]

    guesses = []
    for example in tqdm(examples):
        document = example["content"]
        question = example["query"]
        answer = example["answer"]
        predictions = predictor.predict(document, question, top_n=1)
        prediction = predictions[0][0]

        gs = guesser.guess_single(example["query"])
        gs = sorted(gs.items(), key=lambda x: x[1])[::-1]
        guess = gs[0][0].replace("_", " ")

        guesses.append((prediction, guess, example["answer"]))

    with open("results.pkl", "wb") as f:
        pickle.dump(guesses, f)


def foo():
    dataset = BonusPairsDataset()
    with open("output.scores.lg.txt") as f:
        scores = []
        for line in f.readlines():
            scores.append(float(line))
    print(len(dataset.examples), len(scores))
    selected = []
    curr_qnum = dataset.examples[0]["qnum"]
    curr_ans = dataset.examples[0]["answer"]
    curr_scores = []
    curr_pairs = []
    for example, score in tqdm(zip(dataset.examples, scores)):
        if example["qnum"] == curr_qnum and example["answer"] == curr_ans:
            curr_scores.append(score)
            curr_pairs.append(example)
        else:
            curr_pairs = sorted(list(zip(curr_pairs, curr_scores)), key=lambda x: x[1])[
                ::-1
            ]
            selected.append(curr_pairs[0][0])
            curr_pairs = [example]
            curr_scores = [score]
            curr_qnum = example["qnum"]
            curr_ans = example["answer"]
    with open("selected.pkl", "wb") as f:
        pickle.dump(selected, f)


def bar():

    dataset = BonusPairsDataset()
    examples = [x for x in dataset.examples if x["start"] != -1]
    correct = 0
    for example in tqdm(examples):
        guesses = guesser.guess_single(example["query"])
        guesses = sorted(guesses.items(), key=lambda x: x[1])[::-1]
        guess = guesses[0][0].replace("_", " ")
        correct += guess == example["answer"]
    print(len(examples), correct)


test()
