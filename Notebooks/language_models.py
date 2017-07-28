import gensim
import os
import string
import itertools
from nltk.corpus import brown, movie_reviews, treebank, webtext, gutenberg
import sense2vec
from operator import itemgetter
import numpy as np
import pickle

class Model:
    def __init__(self, model_type):
        self.model_type = model_type

    def get_type(self):
        return self.model_type

    def similarity(self):
        raise NotImplementedError

    def format_word(self):
        raise NotImplementedError

    def in_vocab(self):
        raise NotImplementedError

class Sense2VecModel(Model):
    def __init__(self, model_type='s2v', model_size=None):
        Model.__init__(self, model_type)
        self.model = sense2vec.load()
        self.model_size = model_size
        self.token_count = self.count_tokens()
        self.pos_list_dict = {}

    def save(self):
        # need to self_save the internal workings!
        file_name = self.model_type + ('_' + self.model_size if self.model_size is not None else '') 
        self.model.save(file_name) #requires a directory called file_name

    def load(self):
        # need to self_load the internal workings!
        try:
            file_name = self.model_type + ('_' + self.model_size if self.model_size is not None else '') 
            self.model = sense2vec.load(file_name)
        except:
            self.model = None

    def count_tokens(self):
        print("counting tokens")
        return sum([v[0] for k,v in self.model.items() if v[0]])

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

    def most_frequent_POS(self, word):
        untagged_word = word.split('|')[0].lower()
        # select highest frequency version
        # return the tagged version of the untagged_word
        freq_list = sorted(all_pos_freq(untagged_word), key=itemgetter(1), reverse=True)
        return freq_list[0][0]

    def pos_list(self, word):
        # returns the list of all the known POS versions of word (from model)
        try:
            retval = self.pos_list_dict[word.split('|')[0].lower()]
        except: 
            retval = list(set([w.split('|')[0].lower()+'|'+w.split('|')[1] for w,f in self.all_word_pos(word)]))
            self.pos_list_dict[word.split('|')[0].lower()] = retval
        return retval
#        return list(set([w.split('|')[0].lower()+'|'+w.split('|')[1] for w,f in self.all_word_pos(word)]))

    def all_word_pos(self, word):
        # get all the known tags for untagged word
        untagged_word = word.split('|')[0].lower()
        return [(key,value[0]) for key, value in self.model.items() if key.lower().startswith(untagged_word+'|')]

    def all_pos_freq(self, word):
        return [(p,sum([f for w,f in self.all_word_pos(word) if w.split('|')[0].lower()+'|'+w.split('|')[1] == p])) for p in self.pos_list(word)]

    def in_vocab(self, word):
        return (word in self.model)

    def probability(self, word, all_pos=False):
        # requires sum of fequencies for full vocab
        if all_pos:
        # requires sum of frequencies for all_pos of a word
            freq = sum([f for w,f in self.all_pos_freq(word) if f])
        else:
        # requires frequency of word
            freq = self.model[word][0]
        return freq/self.token_count

    def score(self, tagged_string):
        return sum(np.log([self.probability(a) for a in tagged_string.split() if self.in_vocab(a)]))

    def entropy(self, tagged_string):
        '''
        this is intended to calculate the entropy of the string
            entropy = -sum_i(prob(w_i)*log(prob(w_i)))

        tagged_string: string
                     : in the form "word1|POS word2|POS ..."
        '''
        return -sum([self.probability(a) * np.log(self.probability(a)) for a in tagged_string.split() if self.in_vocab(a)])

class Word2VecModel(Model):
    def __init__(self, model_type='w2v', model_size='full'):
        Model.__init__(self, model_type)
        self.model_size = model_size
        if model_size == 'full':
            self.model_h = gensim.models.Word2Vec(brown.sents()+movie_reviews.sents()+treebank.sents()+webtext.sents()+gutenberg.sents(), hs=1, negative=0)
            self.model_n = gensim.models.Word2Vec(brown.sents()+movie_reviews.sents()+treebank.sents()+webtext.sents()+gutenberg.sents())
        elif model_size == 'webtext':
            self.model_h = gensim.models.Word2Vec(webtext.sents(), hs=1, negative=0)
            self.model_n = gensim.models.Word2Vec(webtext.sents())
        else:
            self.model_h = gensim.models.Word2Vec(treebank.sents(), hs=1, negative=0)
            self.model_n = gensim.models.Word2Vec(treebank.sents())

    def save(self):
        # need to self_save the internal workings!
        file_name = self.model_type + ('_' + self.model_size if self.model_size is not None else '') 
        self.model_h.save(file_name+'.modh')
        self.model_n.save(file_name+'.modn')

    def load(self):
        # need to self_load the internal workings!
        file_name = self.model_type + ('_' + self.model_size if self.model_size is not None else '') 
        try:
            self.model_h = gensim.models.Word2Vec.load(file_name+'.modh')
            self.model_n = gensim.models.Word2Vec.load(file_name+'.modn')
        except:
            # this will throw all kinds of errors maybe this should rebuild?
            self.model_h = self.model_n = None

    def similarity(self, word1, word2):
        return self.model_n.similarity(self.format_word(word1), self.format_word(word2))

    def format_word(self, word):
        # if a POS tag is present, ignore it. force the word to lowercase
        return word.split('|')[0].lower()

    def in_vocab(self, word):
        return (self.format_word(word) in self.model_n)

    def pos_list(self, word):
        # returns the list of all the known POS versions of word (from model)
        # dumb function to return a single POS version of the word (for comaptability)
        return [word.split('|')[0]+'|X']

    def probability(self, word):
        # use the model_h.score to calculate the probability
        return np.exp(self.score(format_word(word)))

    def score(self, tagged_string):
        untagged_string = ' '.join([self.format_word(w) for w in tagged_string.split() if self.in_vocab(self.format_word(w))])
        return self.model_h.score([untagged_string.split()])

    def entropy(self, tagged_string):
        '''
        this is intended to calculate the entropy of the string
            entropy = -sum_i(prob(w_i)*log(prob(w_i)))
        the gensim documentation says score returns the log probability

        tagged_string: string
                     : in the form "word1|POS word2|POS ..."
        '''
        # for w in tagged_string.split():
        #     log_probability = self.score(w)
        #     word_entropy += -(np.exp(log_probability) * log_probability)
        string_entropy = -sum([np.exp(self.score(w))*self.score(w) for w in tagged_string.split()])
        return string_entropy[0]
