###############################################################
#                        RNN THEANO
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

# MATH
import random
from math import e,log,sqrt

# LAYERS
from layer import OneHot,EmbedLayer,LSTMLayer,RecurrentLayer,LinearLayer,SoftmaxLayer

# HELPERS
from wordHelpers import WordHelper
import utils

# INIT RANDOM
srng = RandomStreams()
####################################################################################################
# CONSTANTS

# VARIABLES INIT
X = T.ivector('x')
Y = T.ivector('y')
H = T.matrix('hidden_state')


# LOAD DATA
# New toy problem:
# first couple paragraphs from a random Federalist paper
with open('input.txt','r') as f:
    data = f.read()
f.close()
corpus = data#.lower()
corpus_len = len(corpus)
print("data loaded: {}".format(corpus_len))

# Initialize wordhelper functions
vocab = list(set(corpus))
wh = WordHelper(vocab)

# BATCHES
TRAIN_BATCHES = 1000
TEST_BATCHES = int(TRAIN_BATCHES)# * 0.2)
VALID_BATCHES = int(TRAIN_BATCHES * 0.2)
batch_size = 1 # MAX_WORD_SIZE
embed_size = 100
seq_length = 25
num_batches = int(corpus_len/seq_length)

# TRAINING PARAMS
n_epochs = 100000
cur_epoch = 0
cur_grad = 0.
use_saved = False
# -- decay
decay_epoch = 1000
if len(sys.argv) > 1:
    lr = float(sys.argv[1])
    if len(sys.argv) > 2:
        decay_rate = float(sys.argv[2])
else:
    lr = 0.1
    decay_rate = 1.0#0.33
    
####################################################################################################
# MODEL AND OPTIMIZATION
######################################################################

# RMSprop is for NERDS
def Adagrad(cost, params, mem, lr=1e-1):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p,g,m in zip(params, grads, mem):
        g = T.clip(g,-5.,5)
        new_m = m + (g * g)
        updates.append((m,new_m))
        updates.append((p, p - (lr * g) / T.sqrt(new_m + 1e-8)))
    return updates

##class RNN:
##    def __init__(self,vocab_size,embed_size,output_size,hidden_layer_sizes,batch_size,dropout=None):
##        self.embed_layer = EmbedLayer(vocab_size,embed_size,batch_size)
##        self.batch_size = batch_size
##        self.embed_size = embed_size
##        self.vocab_size = vocab_size
##        self.update_params = self.embed_layer.update_params
##        # The first input of the hidden layers will be embed_size (accepting output from embed matrix)
##        # and the last will be output_size
##        layer_sizes = [embed_size] + hidden_layer_sizes + [output_size]
##        self.hidden_layer_names = []
##        for i in range(len(layer_sizes)-1):
##            name = 'hidden_layer_{}'.format(i)
##            self.hidden_layer_names.append(name)
##            setattr(self,name,LSTMLayer(layer_sizes[i],
##                                       layer_sizes[i+1],
##                                       batch_size,
##                                       name))
##            # Dropout - if provided
##            hl = getattr(self,name)
##            if dropout is not None:
##                hl.dropout = dropout
##            else:
##                hl.dropout = 0
##                
##            # Add the update parameters to the rnn class
##            self.update_params += hl.update_params
##        self.output_layer = SoftmaxLayer(output_size,vocab_size)
##        self.update_params += self.output_layer.update_params
##        
##    def forward_prop(self,X):
##        o = self.embed_layer.forward_prop(X)
##        for layer_name in self.hidden_layer_names:
##            hidden_layer = getattr(self,layer_name)
##            o = hidden_layer.forward_prop(o)
##        self.output_layer.forward_prop(o)
##        return self.output_layer.pred


class RNN:
    def __init__(self,vocab_size,hidden_layer_size,batch_size,dropout=None):
        self.batch_size = batch_size
        self.hidden_size = hidden_layer_size
        self.vocab_size = vocab_size
        # Input Layer
        #self.input_layer = EmbedLayer(vocab_size,embed_size,batch_size)
        #self.input_layer = OneHot(vocab_size,batch_size)
        # Hidden layer
        self.hidden_layer = RecurrentLayer(vocab_size,hidden_layer_size,batch_size,'h')
        # Output Layer
        self.output_layer = SoftmaxLayer(hidden_layer_size,vocab_size)
        # Update Parameters - Backprop
        #self.update_params = self.input_layer.update_params + \
        self.update_params = self.hidden_layer.update_params + \
                             self.output_layer.update_params
        # Memory Parameters for Adagrad
        #self.memory_params = self.input_layer.memory_params + \
        self.memory_params = self.hidden_layer.memory_params + \
                             self.output_layer.memory_params
        
    def forward_prop(self,X,H):
        #o = self.input_layer.forward_prop(X)
        H = self.hidden_layer.forward_prop(X,H)
        pred = self.output_layer.forward_prop(H)
        return pred,H

    def test_hidden(self,X):
        return self.hidden_layer.forward_prop(X)

    
nodes = [100]

# GET DATA
#train_set_x,train_set_y,test_set_x,test_set_y,valid_set_x,valid_set_y = load(hot=False,words=True)

rnn = RNN(wh.vocab_size,nodes[0],batch_size)
params = rnn.update_params
memory_params = rnn.memory_params
y_pred,hidden = rnn.forward_prop(X,H)

cost = T.nnet.categorical_crossentropy(y_pred,Y).sum() 
updates = Adagrad(cost,params,memory_params)

back_prop = theano.function(inputs=[X,Y,H], outputs=[cost,hidden], updates=updates, allow_input_downcast=True,on_unused_input='warn')
predict = theano.function(inputs=[X,H], outputs=[y_pred,hidden], updates=None, allow_input_downcast=True)

#test_updates = theano.function(inputs=[X,Y], outputs=test_back_prop, allow_input_downcast=True,on_unused_input='warn')
print("Model initialized, beginning training")

def predictTest():
    seed = corpus[0]
    output = [seed]
    hidden_state = np.zeros(rnn.hidden_layer.hidden_state_shape)
    for _ in range(100):
        pred_input = wh.id2onehot(wh.char2id(seed)).ravel()
        p,hidden_state = predict(pred_input,hidden_state)
        # Changed from argmax to random_choice - should introduce more variance - good for learnin'
        letter = wh.id2char(np.random.choice(range(wh.vocab_size), p=p.ravel()))
        output.append(letter)
        seed = letter
    print("prediction:",''.join(output))

smooth_loss = -np.log(1.0/wh.vocab_size)*seq_length
n = 0
p = 0
while True:
    if p+1 >= corpus_len or n == 0:
        # Reset memory
        hidden_state = np.zeros(rnn.hidden_layer.hidden_state_shape)
        p = 0 # go to beginning
    c_input = wh.id2onehot(wh.char2id(corpus[p])).ravel()
    c_output = wh.id2onehot(wh.char2id(corpus[p+1])).ravel()
    
    loss,hidden_state = back_prop(c_input,c_output,hidden_state)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if not n % 1000:
        predictTest()
        print("Completed iteration:",n,"Cost: ",smooth_loss,"Learning Rate:",lr)

    p += 1
    n += 1


 
        
print("Training complete")
predictTest()
      
#corpus = [x for x in utils.read_data("/Users/keganrabil/Desktop/text8.zip").split(" ") if x]
#print("Corpus Loaded")
#corpus_len = 1000 #len(corpus)
##for _ in range(1000):
##    total_cost = 0.
##    random.shuffle(corpus)
##    for c in corpus[:corpus_len]:
##        for i in range(len(c)-1):
##            total_cost += back_prop(wh.char2id(c[i]),wh.id2onehot(wh.char2id(c[i+1])))
##        total_cost += back_prop(wh.char2id(c[-1]),wh.id2onehot(wh.EOS))
##    print("Completed iteration:",_,"Cost: ",total_cost/corpus_len)


