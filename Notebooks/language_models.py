import gensim
import os
import string
import itertools
from nltk.corpus import brown, movie_reviews, treebank, webtext, gutenberg
import sense2vec
from operator import itemgetter

class Model:
    def __init__(self, model_type):
        self.model_type = model_type

    def get_type(self):
        return self.model_type

    def similarity(self):
        raise NotImplementedError

    def format_word(self):
        raise NotImplementedError


class Sense2VecModel(Model):
    def __init__(self, model_type='s2v'):
        Model.__init__(self, model_type)
        self.model = sense2vec.load()

    def similarity(self, word1, word2):
        f1,v1 = self.model[self.format_word(word1)]
        f2,v2 = self.model[self.format_word(word2)]
        return self.model.data.similarity(v1,v2)

    def format_word(self, word):
        # if no POS tag is present, find the most freqent version of the word.
        this_word = word.split('|')
        if len(this_word) == 1:
            word = self.most_frequent_POS(this_word[0])
        return word

    def most_frequent_POS(self, untagged_word):
        # get all the known tags for untagged word
        # select highest frequency version
        # return the tagged version of the untagged_word
        freq_list = sorted([(key,value[0]) for key, value in self.model.items() if key.lower().startswith(untagged_word+'|')], key=itemgetter(1), reverse=True)

        return freq_list[0][0]



class Word2VecModel(Model):
    def __init__(self, name, model_type='w2v'):
        Model.__init__(self, model_type)
        self.model_h = gensim.models.Word2Vec(brown.sents()+movie_reviews.sents()+treebank.sents()+webtext.sents()+gutenberg.sents(), hs=1, negative=0)
        self.model_n = gensim.models.Word2Vec(brown.sents()+movie_reviews.sents()+treebank.sents()+webtext.sents()+gutenberg.sents())

    def similarity(self, word1, word2):
        return model.similarity(format_word(word1), format_word(word2))

    def probability(self, sent):
        # use the model_h to calculate the probability?
        raise NotImplementedError
        return 0

    def format_word(self, word):
        # if a POS tag is present, ignore it. force the word to lowercase
        return word.split('|')[0].lower()
