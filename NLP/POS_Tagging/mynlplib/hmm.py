from mynlplib.preproc import conll_seq_generator
from mynlplib.constants import START_TAG, END_TAG, OFFSET, UNK
from mynlplib import naive_bayes, most_common 
import numpy as np
from collections import defaultdict
import torch
import torch.nn
from torch.autograd import Variable
import math


# Deliverable 4.2
def compute_transition_weights(trans_counts, smoothing):
    """
    Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag)] and weights

    weights[(end_tag, start_tag)] = weight of transition from start_tag to end_tag
    """

    tag_sums = {key:sum(value.values()) for (key,value) in trans_counts.items()}
    weights = defaultdict(float)
    normal_tags = list(trans_counts.keys())
    all_tags = normal_tags + [END_TAG]
    valid_ends = normal_tags
    
    for start_tag in [START_TAG] + all_tags:
        for end_tag in [START_TAG] + all_tags:
            if start_tag == END_TAG or end_tag == START_TAG:
                weights[(end_tag, start_tag)] = -np.inf
            else:
                nom = trans_counts[start_tag][end_tag] + smoothing
                denom = tag_sums[start_tag] + (len(all_tags) * smoothing)
                weights[(end_tag, start_tag)] = math.log(nom / denom)
    return weights


# Deliverable 3.2
def compute_weights_variables(nb_weights, hmm_trans_weights, vocab, word_to_ix, tag_to_ix):
    """
    Computes autograd Variables of two weights: emission_probabilities and the tag_transition_probabilties
    parameters:
    nb_weights: -- a dictionary of emission weights
    hmm_trans_weights: -- dictionary of tag transition weights
    vocab: -- list of all the words
    word_to_ix: -- a dictionary that maps each word in the vocab to a unique index
    tag_to_ix: -- a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    
    :returns:
    emission_probs_vr: torch Variable of a matrix of size Vocab x Tagset_size
    tag_transition_probs_vr: torch Variable of a matrix of size Tagset_size x Tagset_size
    :rtype: autograd Variables of the the weights
    """
    # Assume that tag_to_ix includes both START_TAG and END_TAG
    emission_wip = []
    for word in word_to_ix.keys():
        word_weights = []
        for tag in tag_to_ix.keys():
            if tag == START_TAG or tag == END_TAG:
                value = -np.inf
            else:
                value = nb_weights[(tag, word)] if (tag, word) in nb_weights else 0
            word_weights.append(value)
        emission_wip.append(word_weights)
    emission_matrix = torch.tensor(emission_wip)

    trans_wip = []
    # cant transition to START_TAG
    # cant have a start tag of END_TAG
    for end_tag in tag_to_ix.keys():
        trans_weights = []
        for start_tag in tag_to_ix.keys():
            if start_tag == END_TAG or end_tag == START_TAG or (start_tag == START_TAG and end_tag == END_TAG):
                value = -np.inf
            else:
                key = (end_tag, start_tag)
                value = hmm_trans_weights[key] if key in hmm_trans_weights else 0
            trans_weights.append(value)
        trans_wip.append(trans_weights)
    trans_matrix = torch.tensor(trans_wip)
    
    return emission_matrix, trans_matrix
