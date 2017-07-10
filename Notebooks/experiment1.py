# coding: utf-8
# 2017-06-19

import gensim
import os
import string
import itertools
from nltk.corpus import brown, movie_reviews, treebank, webtext, gutenberg
import sense2vec
from operator import itemgetter
from joke_model import JokeModel
from language_models import Sense2VecModel, Word2VecModel

model_choice = 's2v' #['w2v', 's2v']
# for language model operations we can "safely" assume that words are passed as
# 'word|POS' and that Sense2VecModel and Word2VecModel will not break.
# This may not be true of 'word', but I'm working on it.



def load_stopwords(fname='stopwords.txt'):
    stopwords = ['a','to','and','of', 'are', 'she', 'i', 'you', 'is', "i'm", "i'd", 'but', 'so', 'on', 'the', 'me', 'my', 'into', 'be']
    return stopwords

def load_stoptags(fname='stoppos.txt'):
    allpos = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 
            'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'NORP', 
            'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE']
    keeppos = ['ADJ', 'ADV', 'INTJ', 'NOUN',  
            'PROPN', 'SCONJ', 'SYM', 'VERB', 'X', 'NORP', 
            'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE']
    stoppos = list(set(allpos) - set(keeppos))
    return stoppos

def get_similarities(this_model, joke):
    max_sim = -1
    min_sim = 1
    max_words = ()
    min_words = ()
    # joke_words = [word for word in joke.split() if word.split('|')[0].lower() not in stopwords]
    joke_words = [w for w in joke.split() if w.split('|')[0] not in stopwords]
    joke_words = [w for w in joke.split() if w.split('|')[1] not in stoptags]
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

print("Load the jokes")
jokes = JokeModel('jokes.txt')

print("Load stopwords and stoptags")
stopwords = load_stopwords()
stoptags = load_stoptags()

for joke in jokes.tagged_jokes():
    mns, mnw, mxs, mxw = get_similarities(model, joke)
    print(joke)
    print (mns, mnw, mxs, mxw)