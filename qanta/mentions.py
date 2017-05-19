from string import punctuation
from functools import lru_cache
from nltk.tokenize import word_tokenize

import nltk
from nltk.corpus import wordnet as wn


@lru_cache(maxsize=None)
def get_states():
    states = set()
    for ii in wn.synset("American_state.n.1").instance_hyponyms():
        for jj in ii.lemmas():
            name = jj.name()
            if len(name) > 2 and "_" not in name:
                states.add(name)
            elif name.startswith("New_"):
                states.add(name.replace("New_", ""))
    return states


def find_references(sentence, padding=0):
    tags = nltk.pos_tag(word_tokenize(sentence))
    tags.append(("END", "V"))
    states = get_states()

    references_found = []
    this_ref_start = -1
    for i, pair in enumerate(tags):
        word, tag = pair
        if word.lower() == 'this' or word.lower() == 'these':
            this_ref_start = i
        elif all(x in punctuation for x in word):
            continue
        elif word in states:
            continue
        elif this_ref_start >= 0 and tag.startswith('NN') and \
                not tags[i + 1][1].startswith('NN'):
            references_found.append((this_ref_start, i))
            this_ref_start = -1
        elif tag.startswith('V'):
            this_ref_start = -1

    for start, stop in references_found:
        yield (" ".join(x[0] for x in tags[max(0, start - padding):start]).lower(),
               " ".join(x[0] for x in tags[start:stop + 1]).lower(),
               " ".join(x[0] for x in tags[stop + 1:stop + padding + 1]).lower())
