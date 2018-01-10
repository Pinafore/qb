from typing import List
import six
import re
import json
import torch

from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from torchtext.data import Field, RawField, SubwordField, BucketIterator
from torchtext.vocab import Vocab, pretrained_aliases, Vectors


ftp_patterns = {
    '\n',
    ', for 10 points,',
    ', for ten points,',
    '--for 10 points--',
    'for 10 points, ',
    'for 10 points--',
    'for ten points, ',
    'for 10 points ',
    'for ten points ',
    ', ftp,'
    'ftp,',
    'ftp'
}

regex_pattern = '|'.join([re.escape(p) for p in ftp_patterns])
regex_pattern += r'|\[.*?\]|\(.*?\)'

def qb_split(text, nolength_token='nolengthunk'):
    import nltk
    tokens = nltk.word_tokenize(text)
    if len(tokens) == 0:
        return [nolength_token]
    else:
        return tokens


def str_split(text):
    return text.split()

def qb_tokenize(text: str, strip_qb_patterns=True, tokenizer=qb_split) -> List[str]:
    if strip_qb_patterns:
        text = re.sub(
            '\s+', ' ',
            re.sub(regex_pattern, ' ', text, flags=re.IGNORECASE)
        ).strip().capitalize()

    return tokenizer(text)


class LongField(RawField):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return int(x)

    def process(self, batch, **kwargs):
        return torch.LongTensor(batch)




class QBVocab(Vocab):
    def load_vectors(self, vectors):
        if not isinstance(vectors, list):
            vectors = [vectors]
        for idx, vector in enumerate(vectors):
            if six.PY2 and isinstance(vector, str):
                vector = six.text_type(vector)
            if isinstance(vector, six.string_types):
                # Convert the string pretrained vector identifier
                # to a Vectors object
                if vector not in pretrained_aliases:
                    raise ValueError(
                        "Got string input vector {}, but allowed pretrained "
                        "vectors are {}".format(
                            vector, list(pretrained_aliases.keys())))
                vectors[idx] = pretrained_aliases[vector]()
            elif not isinstance(vector, Vectors):
                raise ValueError(
                    "Got input vectors of type {}, expected str or "
                    "Vectors object".format(type(vector)))

        tot_dim = sum(v.dim for v in vectors)
        self.vectors = torch.rand(len(self), tot_dim) * .08 * 2 - .08
        for i, token in enumerate(self.itos):
            start_dim = 0
            for v in vectors:
                end_dim = start_dim + v.dim
                self.vectors[i][start_dim:end_dim] = v[token.strip()]
                start_dim = end_dim

            assert (start_dim == tot_dim)


class QBTextField(Field):
    vocab_cls = QBVocab


s3_url_pattern = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/datasets/non_naqt/quiz-bowl.{fold}.json'

class QuizBowl(Dataset):
    name = 'quizbowl'
    dirname = ''
    urls = [s3_url_pattern.format(fold=fold) for fold in ['train', 'val', 'dev']]


    @staticmethod
    def sort_key(example):
        return len(example.text)

    def __init__(self, path, qnum_field, sent_field, text_field, page_field, example_mode='sentence', **kwargs):
        self.path = path
        self.example_mode = example_mode
        example_fields = {
            'qnum': [('qnum', qnum_field)],
            'sent': [('sent', sent_field)],
            'text': [('text', text_field)],
            'page': [('page', page_field)]
        }

        with open(path) as f:
            examples = []
            for ex in json.load(f)['questions']:
                if example_mode == 'sentence':
                    sentences = ex['sentences']
                    for i, s in enumerate(sentences):
                        examples.append(Example.fromdict({
                            'qnum': ex['qnum'],
                            'sent': i,
                            'text': s,
                            'page': ex['page']
                        }, example_fields))
                elif example_mode == 'question':
                    raise NotImplementedError('Question tokenization is not implemented yet, submit a PR!')
                elif example_mode == 'runs':
                    raise NotImplementedError('Run tokenization is not implemented yet, submit a PR!')
                else:
                    raise ValueError(
                        f"Valid modes are 'sentence', 'question', and 'runs', but '{example_mode}' was given")

        dataset_fields = {
            'qnum': qnum_field,
            'sent': sent_field,
            'text': text_field,
            'page': page_field
        }

        super(QuizBowl, self).__init__(examples, dataset_fields, **kwargs)

    @classmethod
    def splits(cls, qnum_field, sent_field, text_field, page_field,
               example_mode='sentence', root='.data',
               train='quiz-bowl.train.json',
               validation='quiz-bowl.val.json',
               test='quiz-bowl.dev.json',
               **kwargs):
        return super(QuizBowl, cls).splits(
            root=root, train=train, validation=validation, test=test, example_mode=example_mode,
            qnum_field=qnum_field, sent_field=sent_field, text_field=text_field, page_field=page_field, **kwargs
        )

    @classmethod
    def iters(cls, example_mode='sentence', batch_size=128, device=0, root='.data', vectors='glove.6B.300d', **kwargs):
        QNUM = LongField()
        SENT = LongField()
        TEXT = QBTextField(batch_first=True, tokenize=qb_tokenize, include_lengths=True, lower=False)
        PAGE = Field(sequential=False, tokenize=str_split)

        train, val, dev = cls.splits(QNUM, SENT, TEXT, PAGE, root=root, example_mode=example_mode, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        PAGE.build_vocab(train)

        return BucketIterator.splits(
            (train, val, dev),
            batch_size=batch_size,
            device=device,
            repeat=False
        )
