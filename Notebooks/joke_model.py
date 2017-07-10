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

import sense2vec

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

class JokeModel():
    def __init__(self, 
                 joke_file = 'jokes.txt', 
                 noun_chunks = True, 
                 named_entities = True,
                 force_lower = True):
        self.joke_file = joke_file
        self.noun_chunks = noun_chunks
        self.named_entities = named_entities
        self.force_lower = force_lower
        self.nlp = spacy.en.English()

    def raw_jokes(self):
        with open(self.joke_file) as f:
            for line in f:
                line = line.strip()
                if self.force_lower:
                    line = line.lower()
                yield(line)

    def tagged_jokes(self):
        for doc in self.nlp.pipe(self.raw_jokes(), n_threads=4):
            yield(self.transform_doc(doc).strip())


    def transform_doc(self, doc):
        # label named entities
        if self.named_entities:
            for ent in doc.ents:
                ent.merge(ent.root.tag_, ent.text, LABELS[ent.label_])

        # compound noun phrases
        if self.noun_chunks:
            for np in list(doc.noun_chunks):
                while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
                    np = np[1:]
                np.merge(np.root.tag_, np.text, np.root.ent_type_)

        # for each sentence in the current doc, tag the words
        strings = []
        for sent in doc.sents:
            if sent.text.strip():
                strings.append(' '.join(self.represent_word(w) for w in sent if not w.is_space))
        if strings:
            return '\n'.join(strings) + '\n'
        else:
            return ''

    def represent_word(self, word):
        if word.like_url:
            return '%%URL|X'
        text = re.sub(r'\s', '_', word.text)
        tag = LABELS.get(word.ent_type_, word.pos_)
        if not tag:
            tag = '?'
        return text + '|' + tag

    def load_jokes(self, fname='jokes.txt'):
        with open(fname) as f:
            the_jokes = f.readlines()

        # you may also want to remove whitespace characters like `\n` at the end of each line
        the_jokes = [x.strip() for x in the_jokes] 

        # deal with punctuation except apostrophes
        punct = ''.join(c for c in string.punctuation if c not in "'")
        the_jokes = [''.join(c for c in s if c not in punct) for s in the_jokes]
        return the_jokes

