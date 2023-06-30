import operator
from collections import defaultdict, Counter
from mynlplib.preproc import conll_seq_generator
from mynlplib.constants import OFFSET, START_TAG, END_TAG, UNK

# sorts a list of tuples by tag (alphabetically), then flips of tuples from being (TAG, COUNT) -> (COUNT, TAG)
def flip(todo):
    med = list(todo)
    med.sort()
    return list(map(lambda x: (x[1], x[0]), med))

def init_weights(most_common_label, labels):
    output = defaultdict(float)
    output.setdefault((most_common_label, constants.OFFSET), 1)
    for label in [label for label in labels if label != most_common_label]:
        output.setdefault((label, constants.OFFSET), 0)
    return output

# Deliverable 1.1
# DONE
def get_tag_word_counts(trainfile):
    """
    Produce a Counter of occurences of word for each tag
    
    Parameters:
    trainfile: -- the filename to be passed as argument to conll_seq_generator
    :returns: -- a default dict of counters, where the keys are tags.
    """
    all_counters = defaultdict(lambda: Counter())
    file_data = conll_seq_generator(trainfile)
    for (words, tags) in file_data:
        for idx, tag in enumerate(tags):
            all_counters[tag].update([words[idx]])
    return all_counters


def get_tag_to_ix(input_file):
    """
    creates a dictionary that maps each tag (including the START_TAG and END_TAG to a unique index and vice-versa
    :returns: dict1, dict2
    dict1: maps tag to unique index
    dict2: maps each unique index to its own tag
    """
    tag_to_ix={}
    for i,(words,tags) in enumerate(conll_seq_generator(input_file)):
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    
    #adding START_TAG and END_TAG
    #if START_TAG not in tag_to_ix:
    #    tag_to_ix[START_TAG] = len(tag_to_ix)
    #if END_TAG not in tag_to_ix:
    #    tag_to_ix[END_TAG] = len(tag_to_ix)
    
    ix_to_tag = {v:k for k,v in tag_to_ix.items()}
    
    return tag_to_ix, ix_to_tag


def get_word_to_ix(input_file, max_size=100000):
    """
    creates a vocab that has the list of most frequent occuring words such that the size of the vocab <=max_size, 
    also adds an UNK token to the Vocab and then creates a dictionary that maps each word to a unique index, 
    :returns: vocab, dict
    vocab: list of words in the vocabulary
    dict: maps word to unique index
    """
    vocab_counter=Counter()
    for words,tags in conll_seq_generator(input_file):
        for word,tag in zip(words,tags):
            vocab_counter[word]+=1
    vocab = [ word for word,val in vocab_counter.most_common(max_size-1)]
    vocab.append(UNK)
    
    word_to_ix={}
    ix=0
    for word in vocab:
        word_to_ix[word]=ix
        ix+=1
    
    return vocab, word_to_ix


def get_noun_weights():
    """Produce weights dict mapping all words as noun"""
    weights = defaultdict(float)
    weights[('NOUN'),OFFSET] = 1.
    return weights


# Deliverable 2.2
# tagger takes (word, possible_tags)
# weights = <(tag, word), weight>
def get_most_common_word_weights(trainfile):
    """
    Return a set of weights, so that each word is tagged by its most frequent tag in the training file.
    If the word does not appear in the training data,
    the weights should be set so that the tagger outputs the most common tag in the training data.
    For the out of vocabulary words, you need to think on how to set the weights so that you tag them by the most common tag.
    
    Parameters:
    trainfile: -- training file
    :returns: -- classification weights
    :rtype: -- defaultdict

    """
    file_data = conll_seq_generator(trainfile)
    # creates dict<word, tag_counts>
    tags_by_words = defaultdict(lambda: Counter())
    tag_counter = Counter()
    word_set = set()
    word_set.add(OFFSET)
    for (words, tags) in file_data:
        tag_counter.update(tags)
        word_set.update(words)
        for idx, word in enumerate(words):
            tags_by_words[word].update([tags[idx]])
    
    most_common_count, most_common_tag = max(flip(tag_counter.items()))
    weights = defaultdict(lambda: 0.0)
    weights.setdefault((most_common_tag, OFFSET), 1.0)
    # class TagDefault(defaultdict):
    #     def __missing__(self, key):
    #         tag, word = key
    #         if tag == most_common_tag and word not in word_set:
    #             return 1
    #         else:
    #             return 0
    # weights = TagDefault()
    for word, tags in tags_by_words.items():
        tag_sum_word = sum(tags.values())
        for tag in tags:
            weight_offset = 1 if tag != most_common_tag else 0
            weights[(tag, word)] = (tags_by_words[word][tag] / tag_sum_word) + weight_offset

    return weights


# Deliverable 4.1
def get_tag_trans_counts(trainfile):
    """compute a dict of counters for tag transitions

    :param trainfile: name of file containing training data
    :returns: dict, in which keys are tags, and values are counters of succeeding tags
    :rtype: dict

    """
    
    tot_counts = defaultdict(lambda : Counter())
    file_data = conll_seq_generator(trainfile)
    for (words, tags) in file_data:
        complete_tags = [START_TAG] + tags
        lim = len(complete_tags)
        for idx, tag in enumerate(complete_tags):
            next_tag = complete_tags[idx + 1] if idx < lim - 1 else END_TAG
            tot_counts[tag].update([next_tag])
    return dict(tot_counts)
