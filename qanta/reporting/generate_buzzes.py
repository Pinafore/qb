from collections import namedtuple
import click
from functional import seq


Prediction = namedtuple('Prediction', ['score', 'question', 'sentence', 'token'])
Meta = namedtuple('Meta', ['question', 'sentence', 'token', 'guess'])


def load_predictions(pred_file):
    def parse_line(line):
        tokens = line.split()
        score = float(tokens[0])
        q_tokens = [int(x) for x in tokens[1].split('_')]
        return Prediction(score, *q_tokens)
    return seq.open(pred_file).map(parse_line).cache()


def load_meta(meta_file):
    def parse_line(line):
        tokens = line.split('\t')
        question = int(tokens[0])
        sentence = int(tokens[1])
        token = int(tokens[2])
        guess = tokens[3].strip()
        return Meta(question, sentence, token, guess)
    return seq.open(meta_file).map(parse_line).cache()


def load_data(pred_file, meta_file):
    pred = load_predictions(pred_file)
    meta = load_meta(meta_file)


@click.command()
def cli():
    pass


if __name__ == '__main__':
    cli()
