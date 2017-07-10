# coding: utf-8
# 2017-06-19

import gensim
import os
import string
import itertools
from nltk.corpus import brown, movie_reviews, treebank, webtext, gutenberg
import sense2vec
from operator import itemgetter
import simple_tagger
from language_models import Sense2VecModel, Word2VecModel

model_choice = 's2v' #['w2v', 's2v']
# for language model operations we can "safely" assume that words are passed as
# 'word|POS' and that Sense2VecModel and Word2VecModel will not break.
# This may not be true of 'word', but I'm working on it.


def load_jokes(fname='jokes.txt'):
    with open(fname) as f:
        the_jokes = f.readlines()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    the_jokes = [x.strip() for x in the_jokes] 

    # deal with punctuation except apostrophes
    punct = ''.join(c for c in string.punctuation if c not in "'")
    the_jokes = [''.join(c for c in s if c not in punct) for s in the_jokes]
    return the_jokes

def load_stopwords(fname='stopwords.txt'):
    stopwords = ['a','to','and','of', 'are', 'she', 'i', 'you', 'is', "i'm", "i'd", 'but', 'so', 'on', 'the', 'me', 'my', 'into', 'be']
    return stopwords

def load_stoppos(fname='stoppos.txt'):
    allpos = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 
            'PART PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'NORP', 
            'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE']
    keeppos = ['ADJ', 'ADV', 'INTJ', 'NOUN',  
            'PROPN', 'SCONJ', 'SYM', 'VERB', 'X', 'NORP', 
            'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE']
    stoppos = allpos - stoppos
    return stoppos

def vector(word, this_model):
    freq, vector = this_model[word]
    return vector

def freq(word, this_model):
    freq, vector = this_model[word]
    return freq

def get_similarities(this_model, joke):
    max_sim = -1
    min_sim = 1
    max_words = ()
    min_words = ()
    joke_words = [word for word in joke.split() if word.lower() not in stopwords]
    pairs = list(itertools.combinations(joke_words,2))
    for (left_word,right_word) in pairs:
        if not (left_word == right_word):
            try:
                this_sim = this_model.similarity(left_word, right_word)
                if this_sim < min_sim:
                    min_sim = this_sim
                    min_words = (left_word, right_word)
                if this_sim > max_sim:
                    max_sim = this_sim
                    max_words = (left_word, right_word)
            except:
                # use this to build a stopword list
                print("one of these words is not in vocab: {0}, {1}".format(left_word,right_word))
    return [min_sim, min_words, max_sim, max_words]

def get_polysems(this_model, this_word):
    '''
    This should return set of 'word|POS' for the lower case version of word
    '''

    full_ps_list =[key for key, value in this_model.items() if key.lower().startswith(this_word+'|')]

    # for each member of polysems
        # split into left and right
        # lower the left
        # recombine
    # take a set of them to remove duplicates
    polysems = list(set([p.split('|')[0].lower()+'|'+p.split('|')[1].upper() for p in full_ps_list]))

    return polysems

def transform_texts(texts):
    # Load the annotation models
    nlp = English()
    # Stream texts through the models. We accumulate a buffer and release
    # the GIL around the parser, for efficient multi-threading.
    for doc in nlp.pipe(texts, n_threads=4):
        # Iterate over base NPs, e.g. "all their good ideas"
        for np in doc.noun_chunks:
            # Only keep adjectives and nouns, e.g. "good ideas"
            while len(np) > 1 and np[0].dep_ not in ('amod', 'compound'):
                np = np[1:]
            if len(np) > 1:
                # Merge the tokens, e.g. good_ideas
                np.merge(np.root.tag_, np.text, np.root.ent_type_)
            # Iterate over named entities
            for ent in doc.ents:
                if len(ent) > 1:
                    # Merge them into single tokens
                    ent.merge(ent.root.tag_, ent.text, ent.label_)
        token_strings = []
        for token in tokens:
            text = token.text.replace(' ', '_')
            tag = token.ent_type_ or token.pos_
            token_strings.append('%s|%s' % (text, tag))
        yield ' '.join(token_strings)

print("Load the models")
#model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), '../data/GoogleNews-vectors-negative300.bin'), binary=True)
if model_choice == 'w2v':
    print(">>word2vec - extended corpora")
    model = Word2VecModel(model_choice)
elif model_choice == 's2v':
    print(">>sense2vec - reddit hivemind corpus")
    model = Sense2VecModel(model_choice)
else:
    raise NotImplementedError

for joke in pos_jokes('jokes.text'):
    mns, mnw, mxs, mxw = get_similarities(model, joke)
    print(joke)
    print (mns, mnw, mxs, mxw)