import copy
from collections import Counter

import pandas as pd
import numpy as np

# deliverable 1.1
# DONE
def bag_of_words(text):
    '''
    Count the number of word occurences for each document in the corpus
    Every song is a document
    Return a list of Counters that count each word

    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''
    return Counter(text.split())

# deliverable 1.2
# DONE
def aggregate_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''

    counts = Counter()
    for curr_counter in bags_of_words:
        counts.update(curr_counter)
    # YOUR CODE GOES HERE
    
    return counts

# deliverable 1.3
# DONE
def compute_oov(bow1, bow2):
    '''
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    '''
    first_set = set(bow1.copy())
    second_set = set(bow2)

    return first_set.difference(second_set)


# deliverable 1.4
def prune_vocabulary(training_counts, target_data, min_counts):
    '''
    prune target_data to only words that appear at least min_counts times in training_counts

    :param training_counts: aggregated Counter for training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype: list of Counters, set
    '''
    new_target_data = []
    less_than = []
    for word in training_counts:
        if (training_counts[word] < min_counts):
            less_than.append(word)
    
    for curr_counter in target_data:
        new_counter = Counter(curr_counter)
        for word in curr_counter:
            if (training_counts[word] < min_counts):
                del new_counter[word]
        new_target_data.append(new_counter)

    return new_target_data, compute_oov(training_counts, less_than)

# deliverable 5.1
def make_numpy(bags_of_words, vocab):
    '''
    Convert the bags of words into a 2D numpy array

    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a matrix
    :rtype: numpy array
    '''
     # EACH ROW IS AN INSTANCE
    # EACH POSITION CORRESPONDS TO AN INDEX OF WORD IN THE VOCAB SET
    # EACH NUM IN EACH POSITION IS THE COUNT OF SAID WORD IN THE INSTANCE
    output = np.zeros((len(bags_of_words), len(vocab))) 
    ordered_vocab = sorted(vocab)
    for row_idx, instance in enumerate(bags_of_words):
        for word_idx, word in enumerate(ordered_vocab):
            output[row_idx, word_idx] = instance[word]
    return output


### helper code

def read_data(filename,label='Era',preprocessor=bag_of_words):
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string) for string in df['Lyrics'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())
