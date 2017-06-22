
# coding: utf-8

# 2017-06-19
# Read sentence
# Word2Vec
# With the sentence, calculate it's "Probability"
# 

import gensim
import os

def load_jokes(fname='jokes.txt'):
    with open(fname) as f:
        the_jokes = f.readlines()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    the_jokes = [x.strip() for x in the_jokes] 
    return the_jokes



print("Load the model")
model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), '../data/GoogleNews-vectors-negative300.bin'), binary=True)

print("Load the jokes")
joke_text = load_jokes()


for joke in joke_text:
    joke_words = joke.split()
    for left_word in joke_words[0:-1]:
        for right_word in joke_words[1:]:
            print("{0}-{1}: {2}".format(left_word,right_word,model.similarity(left_word, right_word)))

