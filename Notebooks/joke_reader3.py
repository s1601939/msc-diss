
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
from nltk.corpus import brown, movie_reviews, treebank, webtext, gutenberg

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
    polysems =[key for key, value in this_model.items() if key.lower().startswith(this_word+'|')]
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

print("Load the model")
#model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), '../data/GoogleNews-vectors-negative300.bin'), binary=True)
print(">>Hierarchical Softmax version")
model_hsoftmax = gensim.models.Word2Vec(brown.sents()+movie_reviews.sents()+treebank.sents()+webtext.sents()+gutenberg.sents(), hs=1, negative=0)
print(">>Negative sampling version")
model_negative = gensim.models.Word2Vec(brown.sents()+movie_reviews.sents()+treebank.sents()+webtext.sents()+gutenberg.sents())

print("Load the jokes")
joke_text = load_jokes()

print("Load the stop/oov words")
stopwords = load_stopwords()

for joke in joke_text:
    mns, mnw, mxs, mxw = get_similarities(model_negative, joke)
    print(joke)
    print (mns, mnw, mxs, mxw)