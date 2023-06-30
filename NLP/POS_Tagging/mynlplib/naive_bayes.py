from mynlplib.constants import OFFSET
from mynlplib import clf_base, evaluation, preproc
from math import log

import numpy as np
from collections import defaultdict, Counter

def get_nb_weights(trainfile, smoothing):
    """
    estimate_nb function assumes that the labels are one for each document, where as in POS tagging: we have labels for 
    each particular token. So, in order to calculate the emission score weights: P(w|y) for a particular word and a 
    token, we slightly modify the input such that we consider each token and its tag to be a document and a label. 
    The following helper code converts the dataset to token level bag-of-words feature vector and labels. 
    The weights obtained from here will be used later as emission scores for the viterbi tagger.
    
    inputs: train_file: input file to obtain the nb_weights from
    smoothing: value of smoothing for the naive_bayes weights
    
    :returns: nb_weights: naive bayes weights
    """
    # list of every single word in file as dict, <word, count of 1>
    token_level_docs=[]
    # list of every tag for every word in file
    token_level_tags=[]
    for words,tags in preproc.conll_seq_generator(trainfile):
        token_level_docs += [{word:1} for word in words]
        token_level_tags +=tags
    nb_weights = estimate_nb(token_level_docs, token_level_tags, smoothing)
    
    return nb_weights

# Can copy from A1
def estimate_nb(x_vector, y_vector, smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    # WANT DICT OF FORM OF 
    #   DICT[(WORD, LABEL)] = PROBABILITY
    label_set = set(y_vector)
    label_offsets = make_offset_p_y(y_vector, label_set)
    vocab_set = make_vocab_set(x_vector)
    # the dict of 
    #    <label - dict of <word, log probability>
    log_label_pxy_dict = dict()
    for label in label_set:
        log_label_pxy_dict[label] = estimate_pxy(x_vector, y_vector, label, smoothing, vocab_set)
    # output dict of
    #   <(label, word), prob of given label + prob of word given label>
    output = defaultdict(float)
    for label in label_set:
        for word in vocab_set:
            output[(label, word)] = log_label_pxy_dict[label][word]
        output[(label, OFFSET)] = label_offsets[label]
    return output

# Can copy from A1
def estimate_pxy(x_vector, y_vector, label, smoothing, vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    mega_doc = get_corpus_counts(x_vector, y_vector, label)
    total_word_counts_from_vocab = 0
    for word in vocab:
        total_word_counts_from_vocab =  total_word_counts_from_vocab + mega_doc[word]
    output = defaultdict(float)
    vocab_len = len(vocab)
    for word in vocab:
        output[word] = log((mega_doc[word] + smoothing) / (total_word_counts_from_vocab + (smoothing * vocab_len)))
    # output[OFFSET] = log((0 + smoothing) / (total_word_counts_from_vocab + (smoothing * vocab_len)))
    return output

# Can copy from A1
# x_vector is list of dicts, each containing only 1 word as key
# y_vector is list of labels (tags), 1 per word
def get_corpus_counts(x_vector, y_vector, label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    # print(label)
    # output = Counter()
    # output = defaultdict(int)
    # the indices of the words that 
    indices = [idx for idx, _ in enumerate(y_vector) if y_vector[idx] == label]
    valid_x = [list(x_vector[i].keys())[0] for i in indices]
    output = Counter(valid_x)
    return output

# Can copy from A1
def find_best_smoother(x_tr, y_tr, x_dv, y_dv, smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value, scores of all smoothing values
    :rtype: float, dict

    '''
    my_labels = set(y_tr)
    scores = dict()
    for smoother in smoothers:
        curr_thetas = estimate_nb(x_tr, y_tr, smoother)
        y_hat = clf_base.predict_all(x_dv, curr_thetas, my_labels)
        scores[smoother] = evaluation.acc(y_hat, y_dv)
    return clf_base.argmax(scores), scores


def make_offset_p_y(y_vector, labels):
    '''
    find the prior probability for each label

    :param y_vector: training labels
    :param labels: complete set of possible labels
    :returns: log prior probability for each label
    :rtype: dict

    '''
    output = dict()
    label_counts = Counter(y_vector)
    total_labels = len(y_vector)
    for label in labels:
        output[label] = log(label_counts[label] / total_labels)
    return output

def make_vocab_set(x_vector):
    '''
    creates a superset of all the words in each counter given a list of counters of words

    :param x_vector: the list of word counters
    :returns: superset containing all of the words found in the list of counters of words
    :rtype: set

    '''
    output = set()
    for curr_x in x_vector:
        output.update(curr_x.keys())
    return output
