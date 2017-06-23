
# coding: utf-8

# 2017-06-19
# Read sentence
# Word2Vec
# With the sentence, calculate it's "Probability"
# 

import gensim
import os
import string
import itertools

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
    stopwords = ['a', 'to', 'and', 'of']
    return stopwords

def get_similarities(joke):
    max_sim = -1
    min_sim = 1
    max_words = ()
    min_words = ()
    joke_words = [word for word in joke.split() if word.lower() not in stopwords]
    pairs = list(itertools.combinations(joke_words,2))
    for (left_word,right_word) in pairs:
        if not (left_word == right_word):
            try:
                this_sim = model.similarity(left_word, right_word)
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

print("Load the model")
model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), '../data/GoogleNews-vectors-negative300.bin'), binary=True)

print("Load the jokes")
joke_text = load_jokes()

print("Load the stop/oov words")
stopwords = load_stopwords()

for joke in joke_text:
    mns, mnw, mxs, mxw = get_similarities(joke)
    print(joke)
    print (mns, mnw, mxs, mxw)