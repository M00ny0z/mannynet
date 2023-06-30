import operator
from collections import defaultdict, Counter
from mynlplib.constants import START_TAG,END_TAG, UNK
import numpy as np
import torch
import torch.nn
from torch import autograd
from torch.autograd import Variable

def get_torch_variable(arr):
    # returns a pytorch variable of the array
    torch_var = torch.autograd.Variable(torch.from_numpy(np.array(arr).astype(np.float32)))
    return torch_var.view(1,-1)

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def flip(todo):
    med = list(todo)
    med.sort()
    return list(map(lambda x: (x[1], x[0]), med))

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Deliverable 3.3
def viterbi_step(all_tags, tag_to_ix, cur_tag_scores, transition_scores, prev_scores):
    """
    Calculates the best path score and corresponding back pointer for each tag for a word in the sentence in pytorch, which you will call from the main viterbi routine.
    
    parameters:
    - all_tags: list of all tags: includes both the START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    - cur_tag_scores: pytorch Variable that contains the local emission score for each tag for the current token in the sentence
                       it's size is : [ len(all_tags) ] 
    - transition_scores: pytorch Variable that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    - prev_scores: pytorch Variable that contains the scores for each tag for the previous token in the sentence: 
                    it's size is : [ 1 x len(all_tags) ] 
    
    :returns:
    - viterbivars: a list of pytorch Variables such that each element contains the score for each tag in all_tags for the current token in the sentence
    - bptrs: a list of idx that contains the best_previous_tag for each tag in all_tags for the current token in the sentence
    """
    # for every cell in our current layer
    prev_l_scores = prev_scores[0]
    viterbi_scores = []
    back = []
    for end_tag, end_tag_idx in tag_to_ix.items():
        possibil = dict()
        # for every cell in previous layer
        for start_tag, start_tag_idx in tag_to_ix.items():
            emiss_score = cur_tag_scores[end_tag_idx]
            prev_layer_score = prev_l_scores[start_tag_idx]
            trans_score = transition_scores[end_tag_idx][start_tag_idx]
            possibil[start_tag] = prev_layer_score + trans_score + emiss_score
        max_score = max(flip(possibil.items()))
        if (end_tag == START_TAG or end_tag == END_TAG):
            back.append(tag_to_ix[START_TAG])
        else:
            back.append(tag_to_ix[max_score[1]])
        viterbi_scores.append(max_score[0])
    return viterbi_scores, back
#hello world

# Deliverable 3.4
def build_trellis(all_tags, tag_to_ix, cur_tag_scores, transition_scores):
    """
    This function should compute the best_path and the path_score. 
    Use viterbi_step to implement build_trellis in viterbi.py in Pytorch.
    
    parameters:
    - all_tags: a list of all tags: includes START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag to a unique id.
    - cur_tag_scores: a list of pytorch Variables where each contains the local emission score for each tag for that particular token in the sentence, len(cur_tag_scores) will be equal to len(words)
                        it's size is : [ len(words in sequence) x len(all_tags) ] 
    - transition_scores: pytorch Variable (a matrix) that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    
    :returns:
    - path_score: the score for the best_path
    - best_path: the actual best_path, which is the list of tags for each token: exclude the START_TAG and END_TAG here.
    
    Hint: Pay attention to the dimension of cur_tag_scores. It's slightly different from the one in viterbi_step().
    """
    M = len(cur_tag_scores)
    ix_to_tag = { v:k for k,v in tag_to_ix.items() }
    
    # setting all the initial score to START_TAG
    # remember that END_TAG is in all_tags
    initial_vec = np.full((1,len(all_tags)),-np.inf)
    initial_vec[0][tag_to_ix[START_TAG]] = 0
    prev_scores = torch.autograd.Variable(torch.from_numpy(initial_vec.astype(np.float32))).view(1,-1)
    whole_bptrs = []
    for m in range(M):
        viterbi_vars, backptrs = viterbi_step(all_tags, tag_to_ix, cur_tag_scores[m], transition_scores, prev_scores)
        whole_bptrs.append(backptrs)
        prev_scores = get_torch_variable(viterbi_vars)
        path = []
    actual_tags = [tag for tag in all_tags if not (tag == START_TAG or tag == END_TAG)]
    end_tag_choices = []
    for k in actual_tags:
        trans_val = transition_scores[tag_to_ix[END_TAG]][tag_to_ix[k]]
        prev_val = prev_scores[0][tag_to_ix[k]]
        end_tag_choices.append((trans_val + prev_val, k))
    final_score, final_tag = list(max(end_tag_choices))
    
    curr_tag = tag_to_ix[final_tag]
    path.append(ix_to_tag[curr_tag])
    for i in range(M - 1):
        curr_tag = whole_bptrs[m - i][curr_tag]
        path.append(ix_to_tag[curr_tag])
    path.reverse()
    return final_score, path
