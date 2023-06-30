from mynlplib.constants import OFFSET
from mynlplib.constants import BINS
import numpy as np
import torch

# deliverable 6.1
# DONE
def get_top_features_for_label(weights, label, k=5):
    '''
    Return the five features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in 
    :returns: list of tuples of features and weights
    :rtype: list
    '''

    # grabs only the weights that correspond to the given label
    weights_by_given_label = { (curr_label, word): weights[(curr_label, word)] for (curr_label, word) in weights.keys() if curr_label == label}
    top_features = sorted(weights_by_given_label, key=lambda curr_key: weights_by_given_label[curr_key], reverse=True)
    return [(key, weights_by_given_label[key]) for key in top_features[:k]]

# deliverable 6.2
# DONE
def get_top_features_for_label_torch(model, vocab, label_set, label, k=5):
    '''
    Return the five words with the highest weight for a given label.

    :param model: PyTorch model
    :param vocab: vocabulary used when features were converted
    :param label_set: set of ordered labels
    :param label: the label you are interested in 
    :returns: list of words
    :rtype: list
    '''

    # WE CAN GRAB A LAYER FROM A MODEL BY USING REGULAR INDEX
    # layer_weights HAS 4 ARRAYS, EACH CORRESPONDING TO A LABEL
    # EACH INDEX IN ARRAY CORRESPONDS TO WEIGHT OF WORD
    vocab = sorted(vocab)
    layer_weights = model[0].weight
    wanted_label_weights = layer_weights[label_set.index(label)]
    sorted_indices = sorted(range(len(wanted_label_weights)), reverse=True, key = lambda idx: wanted_label_weights[idx])
    return [vocab[idx] for idx in sorted_indices[:k]]

# deliverable 7.1
# DONE
def get_token_type_ratio(instance):
    '''
    compute the ratio of tokens to types

    :param counts: list of bag of words feature for a song, as a numpy array
    :returns: ratio of tokens to types, per instance
    :rtype: float

    '''
    instance_sum = sum(instance)
    if (instance_sum == 0):
        return 0
    tokens = 0
    for count in instance:
        if count > 0:
            tokens = tokens + 1
    return instance_sum / tokens

# deliverable 7.2
def concat_ttr_binned_features(data):
    '''
    Discretize your token-type ratio feature into bins.
    Then concatenate your result to the variable data

    :param data: Bag of words features (e.g. X_tr)
    :returns: Concatenated feature array [Nx(V+7)]
    :rtype: numpy array

    '''
    row_size = len(data[0])
    output = []
    for idx, instance in enumerate(data):
        bins = np.zeros(BINS)
        bin_num = int(get_token_type_ratio(instance))
        if bin_num <= 0:
            bins[0] = 1
        elif bin_num >= 6:
            bins[6] = 1
        else:
            bins[bin_num] = 1
        output.append(np.concatenate((instance, bins)))
    return np.array(output)

    
    raise NotImplementedError
