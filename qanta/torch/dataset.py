import os
import re
import json
import torch

from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from torchtext.data import Field, RawField, BucketIterator
from torchtext.vocab import Vocab, pretrained_aliases, Vectors
from torchtext.utils import download_from_url

from qanta.wikipedia.cached_wikipedia import extract_wiki_sentences

DS_VERSION = '2018.04.18'


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


def create_qb_tokenizer(
        unigrams=True, bigrams=False, trigrams=False,
        zero_length_token='zerolengthunk', strip_qb_patterns=True):
    def tokenizer(text):
        if strip_qb_patterns:
            text = re.sub(
                '\s+', ' ',
                re.sub(regex_pattern, ' ', text, flags=re.IGNORECASE)
            ).strip().capitalize()
        import nltk
        tokens = nltk.word_tokenize(text)
        if len(tokens) == 0:
            return [zero_length_token]
        else:
            ngrams = []
            if unigrams:
                ngrams.extend(tokens)
            if bigrams:
                ngrams.extend([f'{w0}++{w1}' for w0, w1 in nltk.bigrams(tokens)])
            if trigrams:
                ngrams.extend([f'{w0}++{w1}++{w2}' for w0, w1, w2 in nltk.trigrams(tokens)])

            if len(ngrams) == 0:
                ngrams.append(zero_length_token)
            return ngrams

    return tokenizer


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
            if isinstance(vector, str):
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

HTTP_PREFIX = 'http://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets'
S3_URL_PATTERN = os.path.join(HTTP_PREFIX, 'qanta.torchtext.{fold}.2018.04.18.json')
s3_wiki = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/datasets/wikipedia/wiki_lookup.json'


class QuizBowl(Dataset):
    name = 'quizbowl'
    dirname = ''
    urls = [S3_URL_PATTERN.format(fold=fold) for fold in ['train', 'val', 'dev']]


    @staticmethod
    def sort_key(example):
        if hasattr(example, 'text'):
            return len(example.text)
        elif hasattr(example, 'unigram'):
            return len(example.unigram)
        elif hasattr(example, 'bigram'):
            return len(example.bigram)
        elif hasattr(example, 'trigram'):
            return len(example.trigram)
        else:
            raise ValueError('Not valid length fields')

    def __init__(self, path, qanta_id_field, sent_field, page_field,
                 text_field, unigram_field, bigram_field, trigram_field,
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

        text_dependent_fields = []
        if text_field is not None:
            text_dependent_fields.append(('text', text_field))
        if unigram_field is not None:
            text_dependent_fields.append(('unigram', unigram_field))
        if bigram_field is not None:
            text_dependent_fields.append(('bigram', bigram_field))
        if trigram_field is not None:
            text_dependent_fields.append(('trigram', trigram_field))

        example_fields = {
            'qanta_id': [('qanta_id', qanta_id_field)],
            'sent': [('sent', sent_field)],
            'page': [('page', page_field)],
            'text': text_dependent_fields
        }

        examples = []
        answer_set = set()
        with open(path) as f:
            for ex in json.load(f)['questions']:
                if example_mode == 'sentence':
                    sentences = [ex['text'][start:end] for start, end in ex['tokenizations']]
                    for i, s in enumerate(sentences):
                        examples.append(Example.fromdict({
                            'qanta_id': ex['qanta_id'],
                            'sent': i,
                            'text': unidecode(s),
                            'page': ex['page']
                        }, example_fields))
                        answer_set.add(ex['page'])
                elif example_mode == 'question':
                    examples.append(Example.fromdict({
                        'qanta_id': ex['qanta_id'],
                        'sent': -1,
                        'text': unidecode(ex['text']),
                        'page': ex['page']
                    }, example_fields))
                    answer_set.add(ex['page'])
                else:
                    raise ValueError(
                        f"Valid modes are 'sentence' and 'question', but '{example_mode}' was given")

        if use_wiki and n_wiki_sentences > 0 and 'train' in path:
            for page in answer_set:
                if page in self.wiki_lookup:
                    sentences = extract_wiki_sentences(
                        page, self.wiki_lookup[page]['text'], n_wiki_sentences,
                        replace_title_mentions=replace_title_mentions
                    )
                    for i, s in enumerate(sentences):
                        examples.append(Example.fromdict({
                            'qanta_id': -1,
                            'sent': i,
                            'text': s,
                            'page': page
                        }, example_fields))


        dataset_fields = {
            'qanta_id': qanta_id_field,
            'sent': sent_field,
            'page': page_field,
        }
        if text_field is not None:
            dataset_fields['text'] = text_field
        if unigram_field is not None:
            dataset_fields['unigram'] = unigram_field
        if bigram_field is not None:
            dataset_fields['bigram'] = bigram_field
        if trigram_field is not None:
            dataset_fields['trigram'] = trigram_field

        super(QuizBowl, self).__init__(examples, dataset_fields, **kwargs)

    @classmethod
    def splits(cls, example_mode='sentence',
               use_wiki=False, n_wiki_sentences=5, replace_title_mentions='',
               root='.data',
               train=f'qanta.torchtext.train.{DS_VERSION}.json',
               validation=f'qanta.torchtext.val.{DS_VERSION}.json',
               test=f'qanta.torchtext.dev.{DS_VERSION}.json',
               **kwargs):
        remaining_kwargs = kwargs.copy()
        del remaining_kwargs['qanta_id_field']
        del remaining_kwargs['sent_field']
        del remaining_kwargs['page_field']

        remaining_kwargs.pop('text_field', None)
        remaining_kwargs.pop('unigram_field', None)
        remaining_kwargs.pop('bigram_field', None)
        remaining_kwargs.pop('trigram_field', None)
        return super(QuizBowl, cls).splits(
            root=root, train=train, validation=validation, test=test, example_mode=example_mode,
            qanta_id_field=kwargs['qanta_id_field'], sent_field=kwargs['sent_field'], page_field=kwargs['page_field'],
            text_field=kwargs.get('text_field', None),
            unigram_field=kwargs.get('unigram_field', None),
            bigram_field=kwargs.get('bigram_field', None),
            trigram_field=kwargs.get('trigram_field', None),
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
              sort_within_batch=None,
              **kwargs):
        QANTA_ID = LongField()
        SENT = LongField()
        PAGE = Field(sequential=False, tokenize=str_split)
        if combined_ngrams:
            tokenizer = create_qb_tokenizer(unigrams=unigrams, bigrams=bigrams, trigrams=trigrams)
            TEXT = QBTextField(
                batch_first=True,
                tokenize=tokenizer,
                include_lengths=True, lower=lower
            )
            train, val, dev = cls.splits(
                qanta_id_field=QANTA_ID, sent_field=SENT, text_field=TEXT, page_field=PAGE,
                root=root, example_mode=example_mode,
                use_wiki=use_wiki, n_wiki_sentences=n_wiki_sentences, replace_title_mentions=replace_title_mentions,
                **kwargs
            )
            TEXT.build_vocab(train, vectors=vectors, max_size=combined_max_vocab_size)
            PAGE.build_vocab(train)
        else:
            if unigrams:
                unigram_tokenizer = create_qb_tokenizer(unigrams=True, bigrams=False, trigrams=False)
                UNIGRAM_TEXT = QBTextField(
                    batch_first=True, tokenize=unigram_tokenizer,
                    include_lengths=True, lower=lower
                )
            else:
                UNIGRAM_TEXT = None

            if bigrams:
                bigram_tokenizer = create_qb_tokenizer(unigrams=False, bigrams=True, trigrams=False)
                BIGRAM_TEXT = QBTextField(
                    batch_first=True, tokenize=bigram_tokenizer,
                    include_lengths=True, lower=lower
                )
            else:
                BIGRAM_TEXT = None

            if trigrams:
                trigram_tokenizer = create_qb_tokenizer(unigrams=False, bigrams=False, trigrams=True)
                TRIGRAM_TEXT = QBTextField(
                    batch_first=True, tokenize=trigram_tokenizer,
                    include_lengths=True, lower=lower
                )
            else:
                TRIGRAM_TEXT = None

            train, val, dev = cls.splits(
                qanta_id_field=QANTA_ID, sent_field=SENT, page_field=PAGE,
                unigram_field=UNIGRAM_TEXT, bigram_field=BIGRAM_TEXT, trigram_field=TRIGRAM_TEXT,
                root=root, example_mode=example_mode, use_wiki=use_wiki, n_wiki_sentences=n_wiki_sentences,
                replace_title_mentions=replace_title_mentions, **kwargs
            )
            if UNIGRAM_TEXT is not None:
                UNIGRAM_TEXT.build_vocab(train, vectors=vectors, max_size=unigram_max_vocab_size)
            if BIGRAM_TEXT is not None:
                BIGRAM_TEXT.build_vocab(train, max_size=bigram_max_vocab_size)
            if TRIGRAM_TEXT is not None:
                TRIGRAM_TEXT.build_vocab(train, max_size=trigram_max_vocab_size)
            PAGE.build_vocab(train)

        return BucketIterator.splits(
            (train, val, dev),
            batch_size=batch_size,
            device=device,
            repeat=False,
            sort_within_batch=sort_within_batch
        )
