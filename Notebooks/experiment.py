# coding: utf-8
# Experimental code for Stephen Graham's Master's Thesis
# Code Date 31 July 2017
# Exported from jupyter notebook... expect weirdness!
# Hand modified
# Import all the things we might need.

# In[ ]:
import gensim
import os
import string
import itertools
import sense2vec
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from operator import itemgetter
from joke_model import JokeModel
from language_models import Sense2VecModel, Word2VecModel
from nltk.corpus import brown, movie_reviews, treebank, webtext, gutenberg
from nltk.corpus import stopwords
from adjustText import adjust_text # https://github.com/Phlya/adjustText
from scipy import interpolate
from collections import defaultdict

# define the results array parts with "constants"
class RESULT:
    PlainText, TaggedText, TaggedWords, SimGrid, EntGrid, MinEnt, MaxEnt = range(7)


# DEBUGGING
_VERBOSE = True


def load_stopwords(fname='stopwords.txt'):
    # stopwords =  ['a','to','of', 'so', 'on', 'the', 'into']
    # stopwords += ['i', 'me', 'my', 'you', 'us', 'we', 'them', 'she', 'her', 'he', 'him']
    # stopwords += ['and', 'or', 'but']
    # stopwords += ['had', 'have', "'ve"]
    # stopwords += ['is', 'are', 'am', "'m", 'be']
    # stopwords += ["'s", "'d"]
    stopWords = set(stopwords.words('english')) | {'would'}
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


def get_similarities(this_model, this_joke):
    # probably want to make these global so it doesn't have to do this for EVERY joke
    # or put them in the model?
    stop_words = load_stopwords()
    stop_tags = load_stoptags()

    # remove stopwords
    joke_words = [w for w in this_joke.split() if w.split('|')[0].lower() not in stop_words]
    # remove unwanted tags
    joke_words = [w for w in joke_words if w.split('|')[1] not in stop_tags]
    # remove OOV words
    joke_words = [w for w in joke_words if this_model.in_vocab(w)]

    sim_grid = pd.DataFrame(index=joke_words, columns=joke_words)
    #sim_grid = sim_grid.fillna(-1.0)

    pairs = list(itertools.combinations(joke_words,2))
    for (left_word,right_word) in pairs:
        try:
            this_sim = this_model.similarity(left_word, right_word)
            sim_grid[left_word][right_word] = this_sim
            sim_grid[right_word][left_word] = this_sim
        except:
            # we could use this to build a stopword list
            # or we could use ContVec? to reconstruct a new vector for the OOV word?
            print("one of these words is not in vocab: {0}, {1}".format(left_word,right_word))
    return [sim_grid, this_joke, joke_words]


def rank_similarities(this_joke, ascending=True):
    sim_list = sorted([(this_joke[RESULT.SimGrid][x][y],'{} {}'.format(x,y)) 
                   for xi,x in enumerate(this_joke[RESULT.TaggedWords])
                   for yi,y in enumerate(this_joke[RESULT.TaggedWords]) if xi < yi])
    return sim_list


def meandiff_similarities(this_joke):
    sim_grid = this_joke[RESULT.SimGrid].replace(-1,np.nan).mean()
    sim_grid = np.abs(this_joke[RESULT.SimGrid].replace(-1,np.nan) - sim_grid.mean())
    sim_list = sorted([(sim_grid[x][y],'{} {}'.format(x,y)) 
                       for xi,x in enumerate(this_joke[RESULT.TaggedWords])
                       for yi,y in enumerate(this_joke[RESULT.TaggedWords]) if xi < yi])
    return sim_list


def get_entropies(pos_tagged_list):
    entropies = [model.entropy(string) for string in pos_tagged_list]
#     print(entropies)
#     print(np.cumsum(entropies))
    return list(entropies)


def norm_ent_vec(entropy_list):
    '''
    entropy_list is a list-like of pointwise entropy measures
    returns an np.array of normalised pointwise entropy measures
    '''
    return np.array(entropy_list) / np.sum(entropy_list)


def delta_ent_vec(entropy_list_j, entropy_list_a, absolute=False):
    '''
    entropy_list_j is a list-like of pointwise entropy measures (for the joke text)
    entropy_list_a is a list-like of pointwise entropy measures (for the non_joke text)
    returns an np.array of normalised pointwise entropy differences
    '''
    vector = norm_ent_vec(entropy_list_j) - norm_ent_vec(entropy_list_a)
    if absolute: vector = np.absolute(vector)
    return vector


def delta_sim_text_vec(similarity_grid):
    '''
    similarity_grid is an array-like (pandas DataGrid)
    returns an np.array of pointwise similarity range differences
    '''
    return np.array(similarity_grid.replace(-1,np.nan).max(axis=1) - similarity_grid.replace(-1,np.nan).min(axis=1))


def delta_sim_vec(similarity_grid_j, similarity_grid_a, absolute=False):
    '''
    similarity_grid_j is an array-like (pandas DataGrid) from the joke text
    similarity_grid_a is an array-like (pandas DataGrid) from the non-joke text
    returns an np.array of pointwise similarity range differences
    '''
    vector = delta_sim_text_vec(similarity_grid_j) - delta_sim_text_vec(similarity_grid_a)
    if absolute: vector = np.absolute(vector)
    return vector

# dismissal strength measures
def dismissal_strength(joke,ablation,absolute=False):
    '''
    joke is a result structure for a selected joke
    ablation is a result structure for the related ablation of a joke
    returns a list of vector of deltas tuples [wordlist, delta_entropy_vec, delta_similarity_vec]
        each tuple is defined as (feature_string, feature_vector)
    This function could be extended as the number of latent humour features are incorporated.
    '''
    dismissal_strength_list = [('TaggedWords', [[t[0],t[1]] for t in zip(joke[RESULT.TaggedWords],ablation[RESULT.TaggedWords])])] 
    dismissal_strength_list += [('delta_E',delta_ent_vec(joke[RESULT.EntGrid], ablation[RESULT.EntGrid],absolute))]
    dismissal_strength_list += [('delta_S',delta_sim_vec(joke[RESULT.SimGrid], ablation[RESULT.SimGrid],absolute))]
    # here is where we would extend the number of vectors returned
    
    return dismissal_strength_list



def load_model(model, model_size=None, recalculate=False, write=True):
    '''
    model_choice: string representing a language model: Sense2VecModel, Word2VecModel
    model_size: model-specific size string
    from_file: string for loading instead of recalculating
    '''
    # These are the current known language model classes
    model_functions = {'Sense2Vec' : Sense2VecModel,
                       'Word2Vec'  : Word2VecModel} 

    # validate model_choice
    if model not in model_functions:
        valid_values = ', '.join([k for k,v in model_functions.items()])
        raise ValueError("model value must be one of the following: '{}'".format(valid_values))
    
    
    file_name = model + ('_' + model_size if model_size is not None else '') + '.pkl'
    if not recalculate:
        try:
            if _VERBOSE: print('Loading {} model from {} ...'.format(model, file_name))
            # load the surface structure
            with open(file_name,'rb') as pkl_file:
                language_model = pickle.load(pkl_file)
            # call the model's load to restore the deep structure
            language_model.load()
            if _VERBOSE: print('Loaded.')
        except:
            recalculate = True
            
    if recalculate:
        if _VERBOSE: print('Building {} model ...'.format(model))
        try:
            language_model = model_functions[model](model, model_size)
        except:
            raise NotImplementedError
        if _VERBOSE: print("Done.")
        if write:
            # save the surface structure
            with open(file_name,'wb') as pkl_file:
                pickle.dump(language_model, pkl_file)
            # call the model's save method to save the deep structure
            language_model.save()
            if _VERBOSE: print('results saved to {}'.format(file_name))

    return language_model


def load_text(text_choice='jokes.txt'):
    try:
        if _VERBOSE: print("Loading text from {}".format(text_choice))
        jokes = JokeModel(text_choice,named_entities=False)
        if _VERBOSE: print("Done.")
    except:
        raise Exception('Could not load file "'+text_choice+'"')
    return jokes


def get_text_metrics(lang_model, joke_model, recalculate=False, write=True):
    '''
    lang_model: Language Model
              : selected language model to use
    joke_model: Joke Model 
              : model_choice
    returns   : results list

    '''
    file_name = lang_model.model_type+'_'+joke_model.joke_file+'.pkl'
    if not recalculate:
        try:
            with open(file_name,'rb') as pkl_file:
                results = pickle.load(pkl_file)
            if _VERBOSE: print('previous results loaded from {}'.format(file_name))
        except:
            recalculate=True
    
    if recalculate:
        results = [[j,None,None,[None],None,None,None] for j in joke_model.raw_jokes()]
        for joke_id, joke in enumerate(joke_model.tagged_jokes()):
            if _VERBOSE: print(results[joke_id][RESULT.PlainText])
            grid, pos_joke, pos_joke_words = get_similarities(lang_model, joke)
            results[joke_id][RESULT.TaggedText] = pos_joke
            results[joke_id][RESULT.TaggedWords] = pos_joke_words
            results[joke_id][RESULT.SimGrid] = grid

            # maybe we don't store this?
            results[joke_id][RESULT.EntGrid] = list([lang_model.entropy(string) for string in pos_joke_words])

            permuted_tagged_sentence = [' '.join(item) for item 
                                        in list(itertools.product(*[lang_model.pos_list(w) for w in pos_joke_words]))]
            pts_sorted = sorted([[lang_model.entropy(p),p] for p in permuted_tagged_sentence])
            results[joke_id][5] = list([lang_model.entropy(string) for string in pts_sorted[0][1].split()]) # minimum entropy version tagged_string list
            results[joke_id][6] = list([lang_model.entropy(string) for string in pts_sorted[-1][1].split()]) # maximum entropy version tagged_string list

        if write:
            with open(file_name,'wb') as pkl_file:
                pickle.dump(results, pkl_file)
            if _VERBOSE: print('results saved to {}'.format(file_name))

    return results

######################
# Plotting Functions #
######################
def plot_dismissal_2d(dismissal_list, dimensions=(1,2), labels={}, save_plot=False):
    '''
    dismissal list is expected to be a list of tuples:
        (feature_string, feature_vector)
    dimensions is a tuple indicating which two vectors will be plotted
        The default is the first two vectors
    labels should be a dict like {'title':'Some Title', 'xlabel':'Some x label', ...}
    Error-check that there are at least two vectors in the list
    Error-check the list type
    '''
    
    fig, ax1 = plt.subplots(figsize=(7,4))
    x_axis = dismissal_list[dimensions[0]]
    y_axis = dismissal_list[dimensions[1]]
    
    ax1.scatter(x_axis[1],y_axis[1])

    # now label everything we know
    xlabel = labels.get('xlabel', x_axis[0])
    ylabel = labels.get('ylabel', y_axis[0])
    suptitle = labels.get('title','dismissal strength intersections of {} and {}'.format(xlabel,ylabel))
    subtitle_joke = labels.get('joke','')
    subtitle_ablated = labels.get('ablated','') 
    subtitle = ('joke: '+subtitle_joke if subtitle_joke else '') + '\n' + ('ablated: '+subtitle_ablated if subtitle_ablated else '')
            
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if subtitle.strip() != '':
        fig.suptitle(suptitle, fontsize=12, fontweight='bold', y=1.05)
        ax1.set_title(subtitle.strip(), fontsize=10)
    else:
        fig.suptitle(suptitle, fontsize=12, fontweight='bold')

    # label the points
    texts = []
    for i, txt in enumerate(dismissal_list[0][1]):
        txt_label = txt[0] if txt[0]==txt[1] else txt[0]+'\n'+txt[1]
        texts.append(ax1.text(x_axis[1][i], y_axis[1][i], txt_label))

    adjust_text(texts, force_text=0.75, force_points=0.75,
                arrowprops=dict(arrowstyle="-|>", 
                                connectionstyle='arc3,rad=0',
                                color='r', alpha=0.5))
    plt.tight_layout()
    
    if save_plot:
        print('---- {} ----'.format(save_plot))
        try:
            fig.savefig(save_plot)
        except:
            print(save_plot + ' not saved.')
    
    plt.show()


def plot_similarities(this_joke,save_plot=False):
    fig, ax1 = plt.subplots(figsize=(7,4))
    
    similarities = this_joke[3].replace(-1,1) # show unit similarity
    for xi, x in enumerate(this_joke[2]):
        for y in range(len(this_joke[2])):
            similarities[x][y] = np.nan if xi > y else similarities[x][y]
    heatmap = ax1.imshow(similarities, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    ax1.set_xticks(ticks=range(len(this_joke[2])))
    ax1.set_xticklabels(this_joke[2], rotation=60,ha='right')
    ax1.set_yticks(ticks=range(len(this_joke[2])))
    ax1.set_yticklabels(this_joke[2])

    plt.colorbar(heatmap)

    plt.title("Pairwise word similarity (cosine similarity of word vectors)")
    if save_plot:
        print('not saved.')
        pass
        # but we need to maybe save the plots in a folder plots/joke_id
    
    plt.show()


def plot_entropy(this_joke,save_plot=False,show_sims=True,unit_height=False):
    # twin scales code suggested by https://matplotlib.org/devdocs/gallery/api/two_scales.html
    # this_joke[4] should contain the word-by-word entropy list
    # this_joke[5] should contain the entropy list for the min entropy"parse"
    # this_joke[6] should contain the entropy list for the max entropy "parse"
    fig, ax1 = plt.subplots(figsize=(7,4))
    
    individual_entropies = this_joke[RESULT.EntGrid]
    cumulative_entropy = list(np.cumsum(this_joke[RESULT.EntGrid]))
    min_pos_entropy = this_joke[5]
    max_pos_entropy = this_joke[6]
        
    if unit_height:
        max_ent = cumulative_entropy[-1]
        individual_entropies /= max_ent
        cumulative_entropy /= max_ent
        min_pos_entropy /= max_ent
        max_pos_entropy /= max_ent
        ax1.set_ylim([-0.1,1.1])
        
    ax1.plot(cumulative_entropy, label='cumulative')
    ax1.plot(individual_entropies, label='jokePOS')
    ax1.plot(min_pos_entropy, label='minPOS')
    ax1.plot(max_pos_entropy, label='maxPOS')
    ax1.set_ylabel('Entropy (nat)')

    if show_sims:
        # plot the similarity ranges on ax2
        maxes = this_joke[RESULT.SimGrid].replace(-1,np.nan).max(axis=1).tolist()
        mins = this_joke[RESULT.SimGrid].replace(-1,np.nan).min(axis=1).tolist()
        averages = this_joke[RESULT.SimGrid].mean().tolist()
        ax2 = ax1.twinx()
        if unit_height:
            ax2.set_ylim([-0.1,1.1])
    #    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax2.plot(maxes,'g^',linestyle='None')
        ax2.plot(mins,'gv',linestyle='None')
        ax2.plot(averages, 'kx', linestyle='dotted')
        ax2.set_ylabel('Min/Max Similarity')


    
    box = ax1.get_position()
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, fancybox=False, shadow=False)

    ax1.set_xticks(ticks=range(len(this_joke[2])))
    ax1.set_xticklabels(this_joke[2], rotation=23, ha='right')


    if save_plot:
        print('not saved.')
        pass
        # but we need to maybe save the plots in a folder plots/joke_id
    
    plt.show()

def display_dismissal(joke_ablation_pairs, joke_results, non_joke_results, 
                      model_name='unknown', absolute=False, save_plot_list=[]):
    for j,a in joke_ablation_pairs:
        the_joke = joke_results[j]
        the_ablation = non_joke_results[a]
        try:
            print('Joke {}: {}'.format(j,the_joke[RESULT.PlainText]))
            absolute_text = 'absolute ' if absolute else ''
            dismissals = dismissal_strength(the_joke, the_ablation, absolute=absolute) # may die here!
            labels = dict(title='Dismissal plot of ' + absolute_text + 
                          'unit-normalised entropy delta vs similarity delta',
                          xlabel=absolute_text + 'delta_E\n(unit normalised entropy by word)',
                          ylabel=absolute_text + 'delta_S\n(difference in word similarity range)',
                          joke=the_joke[RESULT.PlainText],
                          ablated=the_ablation[RESULT.PlainText])

            save_plot = False
            if isinstance(save_plot_list, list):
                if j in save_plot_list: 
                    save_plot = 'dismissal_ES{}_{}_j{}_a{}.pdf'.format('abs' if absolute else '',model_name,j,a)
            else:
                if save_plot_list:
                    save_plot = 'dismissal_ES{}_{}_j{}_a{}.pdf'.format('abs' if absolute else '',model_name,j,a)

            plot_dismissal_2d(dismissals, labels=labels, save_plot=save_plot)
        except ValueError as err:
            # Although I am catching this error here, it needs to be fixed.
            # The error is thrown during dismissal strength calculation because of different 
            # selected text words between joke and ablation.
            # There may be a work-around available after the hypothesis resolution is estimated.
            # (because we will know the edit distance, maybe)
            if 'operands could not be broadcast together' in err.args[0]:
                print("plot not available: joke and ablation vectors are not the same length")
            else:
                print(err)


def net_dismissal_strength(dismissals, dimensions=(1,2)):
    return np.dot(np.array(dismissals[dimensions[0]][1]),np.array(dismissals[dimensions[1]][1]))

######################
# Main execution here

# instatiate the language models
s2v_model = load_model(model='Sense2Vec', recalculate=False)
w2v_model = load_model(model='Word2Vec', model_size='full', recalculate=False)

# Load the jokes and the ablated jokes
jokes = load_text('jokes.txt')
non_jokes = load_text('non_jokes.txt')

# Load the ablations list that matches jokes with non-jokes
with open('ablations.txt','r') as ab_text:
    joke_ablation_pairs = [[int(s)-1 for s in line.split(',')] for line in ab_text if not line.startswith('#')]


# Calculate results for jokes and ablated jokes under both language models
s2v_joke_results = get_text_metrics(s2v_model,jokes,recalculate=False)
s2v_non_joke_results = get_text_metrics(s2v_model,non_jokes,recalculate=True)
w2v_joke_results = get_text_metrics(w2v_model,jokes,recalculate=False)
w2v_non_joke_results = get_text_metrics(w2v_model,non_jokes,recalculate=True)



# For each result set, plot the entropy and the similarity heatmap
# for r in s2v_joke_results:
#     plot_entropy(r,show_sims=True,save_plot=False,unit_height=True)
#     plot_similarities(r,save_plot=False)

# for r in s2v_non_joke_results:
#     plot_entropy(r,show_sims=True,save_plot=False,unit_height=True)
#     plot_similarities(r,save_plot=False)

# for r in w2v_joke_results:
#     plot_entropy(r,show_sims=True,save_plot=False,unit_height=True)
#     plot_similarities(r,save_plot=False)

# for r in w2v_non_joke_results:
#     plot_entropy(r,show_sims=True,save_plot=False,unit_height=True)
#     plot_similarities(r,save_plot=False)
##
# This may be the compressed version of the above section
##
# for res_set in [s2v_joke_results, s2v_non_joke_results, w2v_joke_results, w2v_non_joke_results]:
#     for r in res_set:
#         plot_entropy(r,show_sims=True,save_plot=False,unit_height=True)
#         plot_similarities(r,save_plot=False)


# For each joke/non-joke pair, plot the entropy 
for j,a in joke_ablation_pairs:
    print(j,a)
    plot_entropy(s2v_joke_results[j], save_plot=False, show_sims=True, unit_height=True)
    plot_entropy(s2v_non_joke_results[a], save_plot=False, show_sims=True, unit_height=True)
    plot_entropy(w2v_joke_results[j], save_plot=False, show_sims=True, unit_height=True)
    plot_entropy(w2v_non_joke_results[a], save_plot=False, show_sims=True, unit_height=True)

# plot the similarity heatmap  
for j,a in joke_ablation_pairs:
    print(j,a)
    plot_similarities(s2v_joke_results[j], save_plot=False)
    plot_similarities(s2v_non_joke_results[a], save_plot=False)
    plot_similarities(w2v_joke_results[j], save_plot=False)
    plot_similarities(w2v_non_joke_results[a], save_plot=False)


# Display the dismissal strength plots for each of the language models
display_dismissal(joke_ablation_pairs, s2v_joke_results, s2v_non_joke_results, 
                    model_name='Sense2Vec', absolute=False, save_plot_list=True)

display_dismissal(joke_ablation_pairs, s2v_joke_results, s2v_non_joke_results, 
                    model_name='Sense2Vec', absolute=True, save_plot_list=True)

display_dismissal(joke_ablation_pairs, w2v_joke_results, w2v_non_joke_results, 
                    model_name='Word2Vec', absolute=False, save_plot_list=True)

display_dismissal(joke_ablation_pairs, w2v_joke_results, w2v_non_joke_results, 
                    model_name='Word2Vec', absolute=True, save_plot_list=True)



# calculate ranked dismissal strength
model_results_dict = dict(Sense2Vec=(s2v_joke_results,s2v_non_joke_results),
                          Word2Vec=(w2v_joke_results,w2v_non_joke_results))
dismissal_table = defaultdict(lambda: np.array([0.0,0.0]))
this_dismiss = [None, None] # [None for kk,vv in model_results_dict.items()]

col=0
dismissal_labels = ['j','a','dismissal','joke_text']
for k,result_set in model_results_dict.items():
    print("\nDismissal for {}".format(k))
    unranked_dismissals = []
    for j,a in joke_ablation_pairs:
        the_joke = result_set[0][j]
        the_ablation = result_set[1][a]
        try:
            dismissals = dismissal_strength(the_joke, the_ablation, absolute=True) # may die here!
#             print('{:.6}: {}'.format(net_dismissal_strength(dismissals),the_joke[RESULT.PlainText]))
            unranked_dismissals += [[j, a, 
                                  net_dismissal_strength(dismissals),
                                  the_joke[RESULT.PlainText]]]
        except ValueError as err:
            # Although I am catching this error here, it needs to be fixed.
            # The error is thrown during dismissal strength calculation because of different 
            # selected text words between joke and ablation.
            # There may be a work-around available after the hypothesis resolution is estimated.
            # (because we will know the edit distance, maybe)
            if 'operands could not be broadcast together' in err.args[0]:
#                 if _VERBOSE: print("dismissal not available: joke and ablation vectors are not the same length")
                pass
            else:
                print(err)
    df = pd.DataFrame.from_records(unranked_dismissals,columns=dismissal_labels,coerce_float=True)
    df = df.groupby(['j','joke_text'])['dismissal'].mean().sort_values()
    ranked_dismissals = df.reset_index().values.tolist()
    for r in ranked_dismissals:
        # this will need to be modified if we have more than two language models to consider!!!
        dismissal_table[r[1]] += np.array([r[2] if col == 0 else 0.0, r[2] if col == 1 else 0.0])
    col += 1
# print(dismissal_table)


def build_dismisal_table_for_thesis(dismissal_table):
    print('\\begin{center}')
    print('\\begin{tabular}{ l l l }')
    for j,s,w in sorted([[k,v[0],v[1]] for k,v in dismissal_table.items()], key=lambda sortval: sortval[1]):
        print('\\hline {} & {:.2e} & {:.2e} \\\\'.format(j, s, w))
    print('\\end{tabular}')
    print('\\end{center}')




