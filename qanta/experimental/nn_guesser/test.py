import json
import spacy
import argparse
import chainer
from tqdm import tqdm

from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.util.constants import GUESSER_TRAIN_FOLD, GUESSER_DEV_FOLD

from qanta.experimental.nn_guesser import nets
from qanta.experimental.nn_guesser import nlp_utils

nlp = spacy.load('en')

def setup_model(setup_dir):
    with open(setup_dir) as f:
        args = argparse.Namespace()
        args.__dict__ = json.loads(f.read())
        print(json.dumps(args.__dict__, indent=2))

    with open(args.vocab_path) as f:
        vocab = json.load(f)

    with open(args.answers_path) as f:
        answers = json.load(f)

    n_class = args.n_class
    n_vocab = len(vocab)
    embed_size = 300
    hidden_size = 512
    hidden_dropout = 0.3
    output_dropout = 0.2
    gradient_clipping = 0.25

    print('# vocab: {}'.format(len(vocab)))
    print('# class: {}'.format(n_class))

    # Setup a model
    if args.model == 'dan':
        encoder = nets.DANEncoder(n_vocab, embed_size, hidden_size,
                dropout=hidden_dropout)
    elif args.model == 'rnn':
        encoder = nets.RNNEncoder(1, n_vocab, embed_size, hidden_size)
    model = nets.NNGuesser(encoder, n_class, dropout=output_dropout)

    chainer.serializers.load_npz(args.model_path, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    return model, vocab, answers, args

def main():
    setup_dir = 'result/nn_guesser/args.json'
    model, vocab, answers, args = setup_model(setup_dir)

    questions = QuestionDatabase().all_questions().values()
    questions = [q for q in questions if q.fold == GUESSER_DEV_FOLD]
    percentages = [0.1, 0.25, 0.5, 0.75, 1.0]
    results = [[] for _ in percentages]
    
    for q in tqdm(questions):
        text = nlp(q.flatten_text())
        for i, per in enumerate(percentages):
            t = text[:int(len(text) * per)]
            t = [w.lower_ for w in t if w.is_alpha or w.is_digit]
            xs = nlp_utils.transform_to_array([t], vocab, with_label=False)
            xs = nlp_utils.convert_seq(xs, device=args.gpu, with_label=False)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                prob = model.predict(xs, softmax=True)[0]
            guess = answers[int(model.xp.argmax(prob))]
            results[i].append(guess == q.page)
    for i, rs in enumerate(results):
        print(percentages[i], sum(rs) / len(rs))


if __name__ == '__main__':
    main()
