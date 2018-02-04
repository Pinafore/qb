from typing import List
import os
import six
import re
import json
import torch

from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from torchtext.data import Field, RawField, BucketIterator
from torchtext.vocab import Vocab, pretrained_aliases, Vectors
from torchtext.utils import download_from_url

from qanta.wikipedia.cached_wikipedia import extract_wiki_sentences


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




def str_split(text):
    return text.split()


# Note that these functions don't serialize with pickle, so you have to use dill/cloudpickle instead
def qb_split(text, nolength_token='nolengthunk'):
    import nltk
    tokens = nltk.word_tokenize(text)
    if len(tokens) == 0:
        return [nolength_token]
    else:
        return tokens


def qb_split_bigrams(text, nolength_token='nolengthunk'):
    import nltk
    tokens = nltk.word_tokenize(text)
    if len(tokens) == 0:
        return [nolength_token]
    else:
        text_bigrams = [f'{w0}++{w1}' for w0, w1 in nltk.bigrams(tokens)]
        return tokens + text_bigrams


def qb_tokenize(text: str, strip_qb_patterns=True, tokenizer=qb_split) -> List[str]:
    if strip_qb_patterns:
        text = re.sub(
            '\s+', ' ',
            re.sub(regex_pattern, ' ', text, flags=re.IGNORECASE)
        ).strip().capitalize()

    return tokenizer(text)


def create_qb_tokenizer(unigrams=True, bigrams=False, trigrams=False):
    pass


def qb_tokenize_bigrams(text: str, strip_qb_patterns=True, tokenizer=qb_split_bigrams) -> List[str]:
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
s3_wiki = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/datasets/wikipedia/wiki_lookup.json'

class QuizBowl(Dataset):
    name = 'quizbowl'
    dirname = ''
    urls = [s3_url_pattern.format(fold=fold) for fold in ['train', 'val', 'dev']]


    @staticmethod
    def sort_key(example):
        return len(example.text)

    def __init__(self, path, qnum_field, sent_field, text_field, page_field,
                 example_mode='sentence',
                 use_wiki=False, n_wiki_sentences=3, replace_title_mentions='',
                 **kwargs):
        from unidecode import unidecode

        if use_wiki and 'train' in path:
            base_path = os.path.dirname(path)
            filename = os.path.basename(s3_wiki)
            output_file = os.path.join(base_path, filename)
            if not os.path.exists(output_file):
                download_from_url(s3_wiki, output_file)
            with open(output_file) as f:
                self.wiki_lookup = json.load(f)
        else:
            self.wiki_lookup = {}
        self.path = path
        self.example_mode = example_mode
        example_fields = {
            'qnum': [('qnum', qnum_field)],
            'sent': [('sent', sent_field)],
            'text': [('text', text_field)],
            'page': [('page', page_field)]
        }

        examples = []
        answer_set = set()
        with open(path) as f:
            for ex in json.load(f)['questions']:
                if example_mode == 'sentence':
                    sentences = ex['sentences']
                    for i, s in enumerate(sentences):
                        examples.append(Example.fromdict({
                            'qnum': ex['qnum'],
                            'sent': i,
                            'text': unidecode(s),
                            'page': ex['page']
                        }, example_fields))
                        answer_set.add(ex['page'])
                elif example_mode == 'question':
                    raise NotImplementedError('Question tokenization is not implemented yet, submit a PR!')
                elif example_mode == 'runs':
                    raise NotImplementedError('Run tokenization is not implemented yet, submit a PR!')
                else:
                    raise ValueError(
                        f"Valid modes are 'sentence', 'question', and 'runs', but '{example_mode}' was given")

        if use_wiki and n_wiki_sentences > 0 and 'train' in path:
            for page in answer_set:
                if page in self.wiki_lookup:
                    sentences = extract_wiki_sentences(
                        page, self.wiki_lookup[page]['text'], n_wiki_sentences,
                        replace_title_mentions=replace_title_mentions
                    )
                    for i, s in enumerate(sentences):
                        examples.append(Example.fromdict({
                            'qnum': -1,
                            'sent': i,
                            'text': s,
                            'page': page
                        }, example_fields))


        dataset_fields = {
            'qnum': qnum_field,
            'sent': sent_field,
            'text': text_field,
            'page': page_field
        }

        super(QuizBowl, self).__init__(examples, dataset_fields, **kwargs)

    @classmethod
    def splits(cls, example_mode='sentence',
               use_wiki=False, n_wiki_sentences=5, replace_title_mentions='',
               root='.data',
               train='quiz-bowl.train.json', validation='quiz-bowl.val.json', test='quiz-bowl.dev.json',
               **kwargs):
        remaining_kwargs = kwargs.copy()
        del remaining_kwargs['qnum_field']
        del remaining_kwargs['sent_field']
        del remaining_kwargs['text_field']
        del remaining_kwargs['page_field']
        return super(QuizBowl, cls).splits(
            root=root, train=train, validation=validation, test=test, example_mode=example_mode,
            qnum_field=kwargs['qnum_field'], sent_field=kwargs['sent_field'],
            text_field=kwargs['text_field'], page_field=kwargs['page_field'],
            use_wiki=use_wiki, n_wiki_sentences=n_wiki_sentences, replace_title_mentions=replace_title_mentions,
            **remaining_kwargs
        )

    @classmethod
    def iters(cls, lower=True, example_mode='sentence',
              use_wiki=False, n_wiki_sentences=5, replace_title_mentions='',
              batch_size=128, device=0, root='.data', vectors='glove.6B.300d',
              unigrams=True, bigrams=False, trigrams=False, combined_ngrams=True,
              combined_max_vocab_size=None,
              unigram_max_vocab_size=None, bigram_max_vocab_size=None, trigram_max_vocab_size=None,
              **kwargs):
        QNUM = LongField()
        SENT = LongField()
        TEXT = QBTextField(
            batch_first=True,
            tokenize=qb_tokenize_bigrams if bigrams else qb_tokenize,
            include_lengths=True, lower=lower
        )
        PAGE = Field(sequential=False, tokenize=str_split)

        train, val, dev = cls.splits(
            qnum_field=QNUM, sent_field=SENT, text_field=TEXT, page_field=PAGE,
            root=root, example_mode=example_mode,
            use_wiki=use_wiki, n_wiki_sentences=n_wiki_sentences, replace_title_mentions=replace_title_mentions,
            **kwargs)

        TEXT.build_vocab(train, vectors=vectors, max_size=max_vocab_size)
        PAGE.build_vocab(train)

        return BucketIterator.splits(
            (train, val, dev),
            batch_size=batch_size,
            device=device,
            repeat=False
        )
