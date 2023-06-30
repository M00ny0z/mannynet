from mynlplib.constants import OFFSET
from collections import defaultdict, Counter
import numpy as np

import operator

# hint! use this.
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

def n_argmax(scores):
    items = list(scores.items())
    items.sort()
    print(items)
    return items[np.argmax([i[1] for i in items])][0]    

# argmax = lambda x : max(x.items(),key=operator.itemgetter(1))[0]


# Deliverable 2.1 - can copy from A1
def make_feature_vector(word,label):
    '''
    take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    '''
    output = defaultdict()
    output[(label, word)] = 1
    output[(label, OFFSET)] = 1
    return output
    

# Deliverable 2.1 - can copy from A1
def predict(word, weights, labels):
    '''
    prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    '''
    output = dict()
    # feature_vector = make_feature_vector(word, )
    for label in labels:
        output[label] = compute_score(word, weights, label)
    if (word == "daily"):
        # print(output)
        # print("weight value: " + str(weights[('NOUN', 'daily')]))
        # print(argmax(output))
        # n_argmax([('new', 200), ('a', 1)])
        return n_argmax(output), output
    # print(base_features)
    return argmax(output), output

def predict_all(x,weights,labels):
    '''
    Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array

    '''
    y_hat = [predict(x_i,weights,labels)[0] for x_i in x]
    return y_hat

# example weight = (('PROPN', 'Al'), 1.0)
def compute_score(x, weights, label):
    total = 0
    feature_vector = make_feature_vector(x, label)
    for key in feature_vector.keys():
        total = total + (weights[key] * feature_vector[key])
    if (x == "daily"):
        print(str(feature_vector) + "-total: " + str(total))
    return total
