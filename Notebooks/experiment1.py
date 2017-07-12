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
from nltk.corpus import stopwords
import numpy as np
import pandas as pd


model_choice = 'w2v' #['w2v', 's2v']
# for language model operations we can "safely" assume that words are passed as
# 'word|POS' and that Sense2VecModel and Word2VecModel will not break.
# This may not be true of 'word', but I'm working on it.



def load_stopwords(fname='stopwords.txt'):
    # stopwords =  ['a','to','of', 'so', 'on', 'the', 'into']
    # stopwords += ['i', 'me', 'my', 'you', 'us', 'we', 'them', 'she', 'her', 'he', 'him']
    # stopwords += ['and', 'or', 'but']
    # stopwords += ['had', 'have', "'ve"]
    # stopwords += ['is', 'are', 'am', "'m", 'be']
    # stopwords += ["'s", "'d"]
    stopWords = set(stopwords.words('english'))
    return stopWords

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

    # remove stopwords
    joke_words = [w for w in joke.split() if w.split('|')[0].lower() not in stop_words]
    # remove unwanted tags
    joke_words = [w for w in joke_words if w.split('|')[1] not in stop_tags]
    # remove OOV words
    joke_words = [w for w in joke_words if this_model.in_vocab(w)]

    sim_grid_min = pd.DataFrame(index=joke_words, columns=joke_words)
    sim_grid_min = sim_grid_min.fillna(1)

    sim_grid_max = pd.DataFrame(index=joke_words, columns=joke_words)
    sim_grid_max = sim_grid_max.fillna(-1)

    pairs = list(itertools.combinations(joke_words,2))
    for (left_word,right_word) in pairs:
        if not (left_word == right_word):
            try:
                this_sim = this_model.similarity(left_word, right_word)
                sim_grid_min[leftword][right_word] = min(sim_grid_min[leftword][right_word], this_sim)
                sim_grid_max[leftword][right_word] = max(sim_grid_min[leftword][right_word], this_sim)

                if this_sim < min_sim:
                    min_sim = this_sim
                    min_words = (left_word, right_word)
                if this_sim > max_sim:
                    max_sim = this_sim
                    max_words = (left_word, right_word)
            except:
                # use this to build a stopword list
                print("one of these words is not in vocab: {0}, {1}".format(left_word,right_word))
    return [min_sim, min_words, max_sim, max_words, sim_grid_min, sim_grid_max, joke, joke_words]


print("Load the models")
if model_choice == 'w2v':
    print(">>word2vec - extended corpora")
    model = Word2VecModel(model_choice)
elif model_choice == 's2v':
    print(">>sense2vec - reddit hivemind corpus")
    model = Sense2VecModel(model_choice)
else:
    raise NotImplementedError

print("Load the jokes")
jokes = JokeModel('jokes.txt',named_entities=False)

print("Load stopwords and stoptags")
stop_words = load_stopwords()
stop_tags = load_stoptags()


results = [[j for j in jokes.raw_jokes()],[None],[None],[None],[None]]
joke_id = 0
for joke in jokes.tagged_jokes():
    mns, mnw, mxs, mxw, grid_min, grid_max, pos_joke, pos_joke_words = get_similarities(model, joke)
    results[joke_id][1] += [pos_joke]
    results[joke_id][2] += [pos_joke_words]
    results[joke_id][3] += [grid_min]
    results[joke_id][4] += [grid_max]
    if joke_id == 22:
        print(results[joke_id])
