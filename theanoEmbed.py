###############################################################
#                        LBFGS THEANO
#                        No more fucking around
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
#from six.moves import range
from six.moves.urllib.request import urlretrieve

# SCIPY
import random
from math import e,log,sqrt
import scipy.optimize

# LAYERS
from layer import EmbedLayer,LSTMLayer,LinearLayer,SoftmaxLayer

# HELPERS
import wordHelpers as wh
import utils

# INIT RANDOM
srng = RandomStreams()
####################################################################################################
# CONSTANTS

# VARIABLES INIT
X = T.iscalar('x')
E = T.vector('embedded_X')
#Y = T.iscalar('y')
Y = T.imatrix('y')
LR = T.scalar('learning_rate')

# BATCHES
TRAIN_BATCHES = 1000
TEST_BATCHES = int(TRAIN_BATCHES)# * 0.2)
VALID_BATCHES = int(TRAIN_BATCHES * 0.2)
batch_size = 1 # MAX_WORD_SIZE
embed_size = 256

n_epochs = 50000
cur_epoch = 0
cur_grad = 0.
use_saved = False

####################################################################################################
# MODEL AND OPTIMIZATION
######################################################################

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        g = T.clip(g,-5.,5)
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

class RNN:
    def __init__(self,vocab_size,embed_size,output_size,hidden_layer_sizes,batch_size):
        self.embed_layer = EmbedLayer(vocab_size,embed_size,batch_size)
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.update_params = self.embed_layer.update_params
        # The first input of the hidden layers will be embed_size (accepting output from embed matrix)
        # and the last will be output_size
        layer_sizes = [embed_size] + hidden_layer_sizes + [output_size]
        self.hidden_layer_names = []
        for i in range(len(layer_sizes)-1):
            name = 'hidden_layer_{}'.format(i)
            self.hidden_layer_names.append(name)
            setattr(self,name,LSTMLayer(layer_sizes[i],
                                       layer_sizes[i+1],
                                       batch_size,
                                       name))
            # Add the update parameters to the rnn class
            hl = getattr(self,name)
            self.update_params += hl.update_params
        self.output_layer = SoftmaxLayer(output_size,vocab_size)
        self.update_params += self.output_layer.update_params
        
    def forward_prop(self,X):
        o = self.embed_layer.forward_prop(X)
        for layer_name in self.hidden_layer_names:
            hidden_layer = getattr(self,layer_name)
            o = hidden_layer.forward_prop(o)
        self.output_layer.forward_prop(o)
        return self.output_layer.pred
    
fox = wh.initFox()

nodes = [10]

# GET DATA
#train_set_x,train_set_y,test_set_x,test_set_y,valid_set_x,valid_set_y = load(hot=False,words=True)

rnn = RNN(wh.vocabulary_size,embed_size,wh.vocabulary_size,nodes,batch_size)

y_pred = rnn.forward_prop(X)

cost = T.nnet.categorical_crossentropy(y_pred,Y).mean() #T.mean((y_pred - Y) ** 2)
params = rnn.update_params
updates = RMSprop(cost,params,lr=LR)
test_back_prop = updates[0]

predict = theano.function(inputs=[X], outputs = y_pred, allow_input_downcast=True)
back_prop = theano.function(inputs=[X,Y,LR], outputs=cost, updates=updates, allow_input_downcast=True)

test_updates = theano.function(inputs=[X,Y], outputs=test_back_prop, allow_input_downcast=True,on_unused_input='warn')

# BEGIN TOY PROBLEM
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


corpus = 'the quick brown fox'
corpus_len = len(corpus)

lr = 0.2
decay_epoch = 1000
decay_rate = 0.5

for _ in range(10000):
    total_cost = 0.
    for i in range(corpus_len-1):
        c = corpus[i]
        c_next = corpus[i+1]
        total_cost += back_prop(wh.char2id(c),wh.id2onehot(wh.char2id(c_next)),lr)
    print("Completed iteration:",_,"Cost: ",total_cost/corpus_len,"Learning Rate:",lr)
    if not _ % decay_epoch:
        lr *= decay_rate
        
print("Training complete")
seed = corpus[0]
output = [seed]
for _ in range(len(corpus)-1):
    p = predict(wh.char2id(seed))
    letter = wh.id2char(np.argmax(p,axis=1))
    output.append(letter)
    seed = letter

print("prediction:",output)
      
    
