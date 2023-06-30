import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from mynlplib.constants import UNK, START_TAG, END_TAG
import matplotlib .pyplot as plt
from mynlplib import viterbi
import pickle
from mynlplib import evaluation
from ncps.torch import LTC
from ncps.wirings import AutoNCP

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else to_ix[UNK] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def log_sum_exp(vec):
    # calculates log_sum_exp in a stable way
    max_score = vec[0][argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return (max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast))))

class NCP(nn.Module):
    """
    Class for the NCP model tagger
    """

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, ncp_neurons, embeddings=None):
        super(NCP, self).__init__()

        self.embedding_dim = embedding_dim
        # self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {v:k for k,v in tag_to_ix.items()}
        self.tagset_size = len(tag_to_ix)
        self.ncp_neurons = ncp_neurons

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        if embeddings is not None:
            self.word_embeds.weight.data.copy_(torch.from_numpy(embeddings))
        
        # Maps the embeddings of the word into the hidden state
        self.ncp = LTC(embedding_dim, AutoNCP(ncp_neurons, self.tagset_size), batch_first=False, mixed_memory=True)
        
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.randn(1, self.ncp_neurons)),
                Variable(torch.randn(1, self.ncp_neurons)))
    
    def forward(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        ncp_out, self.hidden = self.ncp(embeds, self.hidden)
        ncp_out = ncp_out.view(len(sentence), -1)
        return ncp_out
    
    def predict(self, sentence):
        """
        this function is used for evaluating the model: 
        Input:
            sentence: a sequence of ids for each word in the sentence
        Outputs:
            Obtains the scores for each token by passing through forward, then passes the scores for each token 
            through a softmax-layer and then predicts the tag with the maximum probability for each token: 
            observe that this is like greedy decoding
        """
        ncp_feats = self.forward(sentence)
        softmax_layer = torch.nn.Softmax(dim=1)
        probs = softmax_layer(ncp_feats)
        idx = argmax(probs)
        tags = [self.ix_to_tag[ix] for ix in idx]
        return tags
    
class CustomBiLSTM(nn.Module):
    """
    Class for the BiLSTM model tagger
    """
    
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, layers, embeddings=None):
        super(CustomBiLSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {v:k for k,v in tag_to_ix.items()}
        self.tagset_size = len(tag_to_ix)
        self.layers = layers
        
        """
        name them as following:
        self.word_embeds: embedding variable
        self.lstm: lstm layer
        self.hidden2tag: fully connected layer
        """
        
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        if embeddings is not None:
            self.word_embeds.weight.data.copy_(torch.from_numpy(embeddings))
        
        # Maps the embeddings of the word into the hidden state
        # we divide hidden_dim by 2 so we can concatenate the forward and backward states together since it is bidirectional
        # that way, together, they are of hidden_dim size
        # for example, if we get passed in a hidden_dim of 30, we set the hidden_size of the lstm to be half of it (15)
        # so that when the lstm gives us the concatented forward and backward features, they add up to 30
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2, num_layers=layers, bidirectional=True)

        # Maps the output of the LSTM into tag space
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_to_ix), bias=True)
        
        
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # axes semantics are: bidirectinal*num_of_layers, minibatch_size, hidden_dimension; we use noisy initialization
        
        return (Variable(torch.randn(2 * self.layers, 1, self.hidden_dim // 2)),
                Variable(torch.randn(2 * self.layers, 1, self.hidden_dim // 2)))
    
    def forward(self, sentence):
        """
        The function obtain the scores for each tag for each of the words in a sentence
        Input:
        sentence: a sequence of ids for each word in the sentence
        Make sure to reshape the embeddings of the words before sending them to the BiLSTM. 
        The axes semantics are: seq_len, mini_batch, embedding_dim
        Output: 
        returns lstm_feats: scores for each tag for each token in the sentence.
        """
        # The PyTorch tutorial on LSTMs suggests something along the following lines model.hidden = model.init_hidden() 
        # You need to clear out the hidden state of the LSTM, detaching it from its history on the last instance.
        # Not doing so will result in not being able to call `loss.backward()` during training
        self.hidden = self.init_hidden()
        # the embedding for each word in the sentence
        # but needs to be of dimensions [words in sentence, 1, embedding_dim] ===> [13, 1, 30]
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # print(lstm_out.size())
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_out = lstm_out.view(len(sentence), -1)
        lstm_feats = self.hidden2tag(lstm_out)
        # print(lstm_out.size())
        return lstm_feats
        
    def predict(self, sentence):
        """
        this function is used for evaluating the model: 
        Input:
            sentence: a sequence of ids for each word in the sentence
        Outputs:
            Obtains the scores for each token by passing through forward, then passes the scores for each token 
            through a softmax-layer and then predicts the tag with the maximum probability for each token: 
            observe that this is like greedy decoding
        """
        lstm_feats = self.forward(sentence)
        softmax_layer = torch.nn.Softmax(dim=1)
        probs = softmax_layer(lstm_feats)
        idx = argmax(probs)
        tags = [self.ix_to_tag[ix] for ix in idx]
        return tags  

class BiLSTM(nn.Module):
    """
    Class for the BiLSTM model tagger
    """
    
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, embeddings=None):
        super(BiLSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {v:k for k,v in tag_to_ix.items()}
        self.tagset_size = len(tag_to_ix)
        
        """
        name them as following:
        self.word_embeds: embedding variable
        self.lstm: lstm layer
        self.hidden2tag: fully connected layer
        """
        
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        if embeddings is not None:
            self.word_embeds.weight.data.copy_(torch.from_numpy(embeddings))
        
        # Maps the embeddings of the word into the hidden state
        # we divide hidden_dim by 2 so we can concatenate the forward and backward states together since it is bidirectional
        # that way, together, they are of hidden_dim size
        # for example, if we get passed in a hidden_dim of 30, we set the hidden_size of the lstm to be half of it (15)
        # so that when the lstm gives us the concatented forward and backward features, they add up to 30
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_to_ix), bias=True)
        
        
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # axes semantics are: bidirectinal*num_of_layers, minibatch_size, hidden_dimension; we use noisy initialization
        
        return (Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                Variable(torch.randn(2, 1, self.hidden_dim // 2)))
    
    def forward(self, sentence):
        """
        The function obtain the scores for each tag for each of the words in a sentence
        Input:
        sentence: a sequence of ids for each word in the sentence
        Make sure to reshape the embeddings of the words before sending them to the BiLSTM. 
        The axes semantics are: seq_len, mini_batch, embedding_dim
        Output: 
        returns lstm_feats: scores for each tag for each token in the sentence.
        """
        # The PyTorch tutorial on LSTMs suggests something along the following lines model.hidden = model.init_hidden() 
        # You need to clear out the hidden state of the LSTM, detaching it from its history on the last instance.
        # Not doing so will result in not being able to call `loss.backward()` during training
        self.hidden = self.init_hidden()
        # the embedding for each word in the sentence
        # but needs to be of dimensions [words in sentence, 1, embedding_dim] ===> [13, 1, 30]
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # print(lstm_out.size())
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_out = lstm_out.view(len(sentence), -1)
        lstm_feats = self.hidden2tag(lstm_out)
        # print(lstm_out.size())
        return lstm_feats
        
    def predict(self, sentence):
        """
        this function is used for evaluating the model: 
        Input:
            sentence: a sequence of ids for each word in the sentence
        Outputs:
            Obtains the scores for each token by passing through forward, then passes the scores for each token 
            through a softmax-layer and then predicts the tag with the maximum probability for each token: 
            observe that this is like greedy decoding
        """
        lstm_feats = self.forward(sentence)
        softmax_layer = torch.nn.Softmax(dim=1)
        probs = softmax_layer(lstm_feats)
        idx = argmax(probs)
        tags = [self.ix_to_tag[ix] for ix in idx]
        return tags



class BiLSTM_CRF(BiLSTM):
    """
    Class for the BiLSTM_CRF model: derived from the BiLSTM class
    """
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, embeddings=None):
        super(BiLSTM_CRF, self).__init__(vocab_size, tag_to_ix, embedding_dim, hidden_dim, embeddings)
        
        """
        adding tag transitions scores as a parameter.
        """
        
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -1000000
        self.transitions.data[:, tag_to_ix[END_TAG]] = -1000000
    
    def forward_alg(self, feats):
        """
        This is the function for the forward algorithm:
        It works very similar to the viterbi algorithm: except that instead of storing just the maximum prev_tag, 
        you sum up the probability to arrive at the curr_tag
        Use log_sum_exp given above to calculate it a numerically stable way.
        
        inputs:
        - feats: -- the hidden states for each token in the input_sequence. 
                Consider this to be the emission potential of each token for each tag.
        - Make sure to use the self.transitions that is defined to capture the tag-transition probabilities
        
        :returns:
        - alpha: -- a pytorch variable containing the score
        """
        
        init_vec = torch.Tensor(1, self.tagset_size).fill_(-1000000)
        # START_TAG has the max score
        init_vec[0][self.tag_to_ix[START_TAG]] = 0.
        
        prev_scores = torch.autograd.Variable(init_vec)
        

        raise NotImplementedError
       
        for feat in feats:
            alphas=[]
            for next_tag in range(self.tagset_size):
                pass

    
    def score_sentence(self, feats, gold_tags):
        """
        Obtain the probability P(x,y) for the labels in tags using the feats and transition_probabilities.
        Inputs:
        - feats: the hidden state scores for each token in the input sentence. 
                Consider this to be the emission potential of each token for each tag.
        - gold_tags: the gold sequence of tags: obtain the joint-log-likelihood score of the sequence 
                    with the feats and gold_tags.
        :returns:
        - a pytorch variable of the score.
        """
        # obtains the score for the sentence for that particular sequence of tags
        score = torch.autograd.Variable(torch.Tensor([0]))
        SENTENCE_LEN = len(feats)
        START_TAG_IDX = self.tag_to_ix[START_TAG]
        END_TAG_IDX = self.tag_to_ix[END_TAG]
        start_trans_val = self.transitions[gold_tags[0]][START_TAG_IDX]
        # adding the START_TAG here
        # tags = torch.cat([Variable(torch.LongTensor([self.tag_to_ix[START_TAG]])), gold_tags])
        FINAL_TAG = gold_tags[SENTENCE_LEN - 1]
        for idx, curr_feats in enumerate(feats):
            curr_tag_idx = gold_tags[idx]
            feat_val = curr_feats[curr_tag_idx]
            trans_val = start_trans_val if idx == 0 else self.transitions[curr_tag_idx][gold_tags[idx - 1]]
            score += feat_val + trans_val
        score += self.transitions[END_TAG_IDX][FINAL_TAG]
        
        return score
    
    def predict(self, sentence):
        """
        This function predicts the tags by using the viterbi algorithm. You should be calling the viterbi algorithm from here.
        Inputs:
        - feats: the hidden state scores for each token in the input sentence. 
                Consider this to be the emission potential of each token for each tag.
        - gold_tags: the gold sequence of tags
        :returns:
        - the best_path which is a sequence of tags
        """
        lstm_feats = self.forward(sentence).view(len(sentence),-1)
        all_tags = [tag for tag,value in self.tag_to_ix.items()]
        
        # call the viterbi algorithm here
        raise NotImplementedError


    def neg_log_likelihood(self, lstm_feats, gold_tags):
        
        """
        This function calculates the negative log-likelihood for the CRF: P(Y|X)
        Inputs: 
        lstm_feats: the hidden state scores for each token in the input sentence. 
        gold_tags: the gold sequence of tags
        :returns:
        score of the neg-log-likelihood for the sentence: 
        You should use the previous functions defined: forward_alg, score_sentence
        """
        
        raise NotImplementedError



def train_model(loss, model, X_tr,Y_tr, word_to_ix, tag_to_ix, X_dv=None, Y_dv = None, num_its=50, status_frequency=10,
               optim_args = {'lr':0.1,'momentum':0},
               param_file = 'best.params'):
    
    #initialize optimizer
    optimizer = optim.SGD(model.parameters(), **optim_args)
    # optimizer = optim.Adam(model.parameters())
    
    losses=[]
    accuracies=[]
    
    for epoch in range(num_its):
        
        loss_value=0
        count1=0
        
        for X,Y in zip(X_tr,Y_tr):
            X_tr_var = prepare_sequence(X, word_to_ix)
            Y_tr_var = prepare_sequence(Y, tag_to_ix)
            
            # set gradient to zero
            optimizer.zero_grad()
            
            lstm_feats= model.forward(X_tr_var)
            output = loss(lstm_feats,Y_tr_var)
            
            output.backward()
            optimizer.step()
            loss_value += output.item()
            count1+=1
            
            
        losses.append(loss_value/count1)
        
        # write parameters if this is the best epoch yet
        acc=0        
        if X_dv is not None and Y_dv is not None:
            acc=0
            count2=0
            for Xdv, Ydv in zip(X_dv, Y_dv):
                
                X_dv_var = prepare_sequence(Xdv, word_to_ix)
                Y_dv_var = prepare_sequence(Ydv, tag_to_ix)
                # run forward on dev data
                Y_hat = model.predict(X_dv_var)
                
                Yhat = np.array([tag_to_ix[yhat] for yhat in Y_hat])
                Ydv = np.array([tag_to_ix[ydv] for ydv in Ydv])
                
                # compute dev accuracy
                acc += (evaluation.acc(Yhat,Ydv))*len(Xdv)
                count2 += len(Xdv)
                # save
            acc/=count2
            if len(accuracies) == 0 or acc > max(accuracies):
                state = {'state_dict':model.state_dict(),
                         'epoch':len(accuracies)+1,
                         'accuracy':acc}
                torch.save(state,param_file)
            accuracies.append(acc)
        # print status message if desired
        if status_frequency > 0 and epoch % status_frequency == 0:
            print("Epoch "+str(epoch+1)+": Dev Accuracy: "+str(acc))
    return model, losses, accuracies
            
    

def plot_results(losses, accuracies):
    fig,ax = plt.subplots(1,2,figsize=[12,2])
    ax[0].plot(losses)
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('iteration');
    ax[1].plot(accuracies);
    ax[1].set_ylabel('dev set accuracy')
    ax[1].set_xlabel('iteration');

    
def obtain_polyglot_embeddings(filename, word_to_ix):
    
    vecs = pickle.load(open(filename,'rb'),encoding='latin1')
    
    vocab = [k for k,v in word_to_ix.items()]
    
    word_vecs={}
    for i,word in enumerate(vecs[0]):
        if word in word_to_ix:
            word_vecs[word] = np.array(vecs[1][i])
    
    word_embeddings = []
    for word in vocab:
        if word in word_vecs:
            embed=word_vecs[word]
        else:
            embed=word_vecs[UNK]
        word_embeddings.append(embed)
    
    word_embeddings = np.array(word_embeddings)
    return word_embeddings
