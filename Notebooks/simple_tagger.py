#merge_text.py
from __future__ import print_function, unicode_literals, division
import io
import bz2
import logging
from toolz import partition
from os import path
import os
import re

import spacy.en
from preshed.counter import PreshCounter
from spacy.tokens.doc import Doc

from joblib import Parallel, delayed
import plac

LABELS = {
    'ENT': 'ENT',
    'PERSON': 'ENT',
    'NORP': 'ENT',
    'FAC': 'ENT',
    'ORG': 'ENT',
    'GPE': 'ENT',
    'LOC': 'ENT',
    'LAW': 'ENT',
    'PRODUCT': 'ENT',
    'EVENT': 'ENT',
    'WORK_OF_ART': 'ENT',
    'LANGUAGE': 'ENT',
    'DATE': 'DATE',
    'TIME': 'TIME',
    'PERCENT': 'PERCENT',
    'MONEY': 'MONEY',
    'QUANTITY': 'QUANTITY',
    'ORDINAL': 'ORDINAL',
    'CARDINAL': 'CARDINAL'
}

def pipe_jokes(fname='jokes.txt'):
    with open(fname) as f:
        for line in f:
            line = line.strip()
            yield(line)


def pos_jokes():
    for doc in nlp.pipe(pipe_jokes(), n_threads=4):
#        tagged_joke = ''
#        for token in doc:                                                                  
#             tagged_joke += '{}|{} '.format(token.text, (token.ent_type or token.pos_))
#        yield(tagged_joke.strip())
#        print(transform_doc(doc))
        yield(transform_doc(doc).strip())


def transform_doc(doc):
    # label named entities
    for ent in doc.ents:
        ent.merge(ent.root.tag_, ent.text, LABELS[ent.label_])
    # compound noun phrases
    for np in list(doc.noun_chunks):
        while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
            np = np[1:]
        np.merge(np.root.tag_, np.text, np.root.ent_type_)
    strings = []
    # for each sentence in the current doc, tag the words
    for sent in doc.sents:
        if sent.text.strip():
            strings.append(' '.join(represent_word(w) for w in sent if not w.is_space))
    if strings:
        return '\n'.join(strings) + '\n'
    else:
        return ''


def represent_word(word):
    if word.like_url:
        return '%%URL|X'
    text = re.sub(r'\s', '_', word.text)
    tag = LABELS.get(word.ent_type_, word.pos_)
    if not tag:
        tag = '?'
    return text + '|' + tag
