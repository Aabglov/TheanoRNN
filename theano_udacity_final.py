###############################################################
#                        THE FINAL UDACITY PROJECT
#                           WORD REVERSER
###############################################################

# THEANO
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# I/O
import pickle
import os
import timeit
import sys
import zipfile
import string
from six.moves.urllib.request import urlretrieve
import copy
import re

# MATH
import random
from math import e,log,sqrt

# LAYERS
from layer import OneHot,EmbedLayer,LSTMLayer,LinearLayer,SoftmaxLayer

# HELPERS
from wordHelpers import WordHelper
import utils

# INIT RANDOM
srng = RandomStreams()
####################################################################################################
# CONSTANTS

# VARIABLES INIT
X_LIST = T.ivector('x_list')
X = T.iscalar('x')

Y_LIST = T.imatrix('y_list')
Y = T.ivector('y')

S = T.dmatrix('hidden_state')
H = T.dmatrix('hidden_update')


# LOAD DATA
# New toy problem:
# first couple paragraphs from a random Federalist paper
with open('input.txt','r') as f:
    data = f.read()
f.close()
corpus = re.split('\W+',data.lower())
corpus_len = len(corpus)
print("data loaded: {}".format(corpus_len))

# Initialize wordhelper functions
vocab = list(set(''.join(corpus)))
wh = WordHelper(vocab)

# BATCHES
batch_size = 1 # MAX_WORD_SIZE
embed_size = 100
seq_length = 8 # average word length
#num_batches = int(corpus_len/seq_length)

# TRAINING PARAMS
n_epochs = 100000
cur_epoch = 0
cur_grad = 0.
use_saved = False
    
####################################################################################################
# MODEL AND OPTIMIZATION
####################################################################################################

# RMSprop is for NERDS
def Adagrad(cost, params, mem, lr=0.1):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p,g,m in zip(params, grads, mem):
        g = T.clip(g,-5.,5)
        new_m = m + (g * g)
        updates.append((m,new_m))
        updates.append((p, p - ((lr * g) / T.sqrt(new_m + 1e-8))))
    return updates

class RNN:
    def __init__(self,vocab_size,embed_size,hidden_layer_size,batch_size,dropout=None):
        self.batch_size = batch_size
        self.hidden_size = hidden_layer_size
        self.vocab_size = vocab_size
        # Input Layer
        self.input_layer = EmbedLayer(vocab_size,embed_size,batch_size)
        # Hidden layer
        self.hidden_layer = LSTMLayer(embed_size,hidden_layer_size,batch_size,'h')
        # Output Layer
        self.output_layer = SoftmaxLayer(hidden_layer_size,vocab_size)
        # Update Parameters - Backprop
        self.update_params = self.input_layer.update_params + \
                             self.hidden_layer.update_params + \
                             self.output_layer.update_params
        # Memory Parameters for Adagrad
        self.memory_params = self.input_layer.memory_params + \
                             self.hidden_layer.memory_params + \
                             self.output_layer.memory_params

    def calc_preds(self,X,Y,S,H):
        e = self.input_layer.forward_prop(X)
        S,H = self.hidden_layer.forward_prop(e,S,H)
        pred = self.output_layer.forward_prop(H)
        return pred,S,H

    def calc_cost(self,pred,Y):
        return T.mean(T.nnet.categorical_crossentropy(pred,Y))
        
    
nodes = [1024]

rnn = RNN(wh.vocab_size,embed_size,nodes[0],batch_size)
params = rnn.update_params
memory_params = rnn.memory_params

outputs_info=[None,dict(initial=S, taps=[-1]),dict(initial=H, taps=[-1])]
y_preds,states,outputs = theano.scan(fn=rnn.calc_preds,
                              outputs_info=outputs_info,
                              sequences=[X_LIST,Y_LIST]
                            )[0] # only need the results, not the updates

scan_costs = theano.scan(fn=rnn.calc_cost,
                              outputs_info=None,
                              sequences=[y_preds,Y_LIST]
                            )[0] # only need the results, not the updates

scan_cost = T.sum(scan_costs)
hidden_state = states[-1]
hidden_output = outputs[-1]
updates = Adagrad(scan_cost,params,memory_params)
back_prop = theano.function(inputs=[X_LIST,Y_LIST,S,H], outputs=[scan_cost,hidden_state,hidden_output], updates=updates)

#grads = T.grad(cost=scan_cost, wrt=params)
#test_grads  = theano.function(inputs=[X_LIST,Y_LIST,H], outputs=grads, updates=None, allow_input_downcast=True)

y_pred = y_preds[-1]
predict = theano.function(inputs=[X_LIST,Y_LIST,S,H], outputs=[y_preds,hidden_state,hidden_output], updates=None, allow_input_downcast=True)

test_hidden = theano.function(inputs=[X_LIST,Y_LIST,S,H], outputs=[states,outputs], updates=None, allow_input_downcast=True)

print("Model initialized, beginning training")

def predictTest():
    test_corpus = corpus[:5]#['the','quick','brown','fox','jumped']
    output = []
    hidden_state = np.zeros(rnn.hidden_layer.hidden_state_shape)
    hidden_output = np.zeros(rnn.hidden_layer.hidden_output_shape)
    for _ in range(5):
        word = test_corpus[_]
        pred_input = []
        pred_output_UNUSED = []
        for i in range(len(word)):
            pred_input.append(wh.char2id(word[i]))
            # This value is only used to trigger the calc_cost.
            # It's incorrect, but it doesn't update the parameters to that's okay.
            # Not great, but okay.
            pred_output_UNUSED.append(wh.id2onehot(wh.char2id(word[i])))
        predictions,hidden_state,hidden_output = predict(pred_input,pred_output_UNUSED,hidden_state,hidden_output)
        for p in predictions:
            letter = wh.id2char(np.random.choice(range(wh.vocab_size), p=p.ravel()))
            output.append(letter)
        output.append(' ')
    print("prediction:",''.join(output),'true:',' '.join(test_corpus))

smooth_loss = -np.log(1.0/wh.vocab_size)*seq_length
n = 0
p = 0
while True:
    if p+seq_length+1 >= corpus_len or n == 0:
        # Reset memory
        hidden_state = np.zeros(rnn.hidden_layer.hidden_state_shape)
        hidden_output = np.zeros(rnn.hidden_layer.hidden_output_shape)
        p = 0 # go to beginning
    c_input = corpus[p]
    c_output = wh.reverseWord(corpus[p])

    batch_input = []
    batch_output = []
    for j in range(len(c_input)):
        c = c_input[j]
        c2 = c_output[j]
        batch_input.append(wh.char2id(c))
        batch_output.append(wh.id2onehot(wh.char2id(c2)))
        
    loss,hidden_state,hidden_output = back_prop(batch_input,batch_output,hidden_state,hidden_output)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if not n % 100:
        predictTest()
        print("Completed iteration:",n,"Cost: ",smooth_loss)

    p += seq_length
    n += 1

        
print("Training complete")
predictTest()
      

