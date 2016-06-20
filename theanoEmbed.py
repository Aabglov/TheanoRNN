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
Y_LIST = T.imatrix('y_list')
X = T.iscalar('x')
Y = T.ivector('y')
LR = T.scalar('learning_rate')



# LOAD DATA
# New toy problem:
# first couple paragraphs from a random Federalist paper
with open('input.txt','r') as f:
    data = f.read()
f.close()
corpus = data.lower()
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
    lr = 0.3
    decay_rate = 1.0#0.33
    
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

# RMSprop is for NERDS
def Adagrad(cost, params, mem, lr=1e-1):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p,g,m in zip(params, grads, mem):
        g = T.clip(g,-5.,5)
        new_m = m + g * g
        updates.append((m,new_m))
        updates.append((p, p - lr * g / np.sqrt(new_m + 1e-8)))
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
        self.input_layer = OneHot(vocab_size,batch_size)
        # Hidden layer
        self.hidden_layer = LSTMLayer(vocab_size,hidden_layer_size,batch_size,'h',dropout)
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
        
##    def forward_prop(self,X):
##        o = self.input_layer.forward_prop(X)
##        o = self.hidden_layer.forward_prop(o)
##        pred = self.output_layer.forward_prop(o)
##        return pred

    def calc_cost(self,X,Y):
        o = self.input_layer.forward_prop(X)
        o = self.hidden_layer.forward_prop(o)
        pred = self.output_layer.forward_prop(o)
        cost = T.nnet.categorical_crossentropy(pred,Y).mean()
        return cost,pred
    
nodes = [100]

# GET DATA
#train_set_x,train_set_y,test_set_x,test_set_y,valid_set_x,valid_set_y = load(hot=False,words=True)

rnn = RNN(wh.vocab_size,nodes[0],batch_size)
params = rnn.update_params
memory_params = rnn.memory_params

##cost = T.nnet.categorical_crossentropy(y_pred,Y).mean() 
##updates = Adagrad(cost,params,memory_params,lr=LR)
##back_prop = theano.function(inputs=[X,Y,LR], outputs=cost, updates=updates, allow_input_downcast=True,on_unused_input='warn')

scan_results,y_preds = theano.scan(fn=rnn.calc_cost,
                              #outputs_info=None,
                              sequences=[X_LIST,Y_LIST]
                            )[0] # only need the results, not the updates

scan_cost = T.sum(scan_results)
updates = Adagrad(scan_cost,params,memory_params,lr=LR)
back_prop = theano.function(inputs=[X_LIST,Y_LIST,LR], outputs=scan_cost, updates=updates)

y_pred = y_preds[-1]
predict = theano.function(inputs=[X_LIST,Y_LIST], outputs=y_pred, allow_input_downcast=True)

#test_updates = theano.function(inputs=[X,Y], outputs=test_back_prop, allow_input_downcast=True,on_unused_input='warn')
print("Model initialized, beginning training")

def predictTest():
    seed = corpus[0]
    output = [seed]
    for _ in range(seq_length):
        pred_input = [wh.char2id(seed)]
        pred_output_UNUSED = [wh.id2onehot(wh.char2id(seed))] # This value is only used to trigger the calc_cost.  It's incorrect, but it doesn't update the parameters to that's okay. Not great, but okay.
        p = predict(pred_input,pred_output_UNUSED)
        # Changed from argmax to random_choice - should introduce more variance - good for learnin'
        letter = wh.id2char(np.random.choice(range(wh.vocab_size), p=p.ravel()))
        output.append(letter)
        seed = letter
    print("prediction:",''.join(output))

for _ in range(n_epochs):
    smooth_loss = -np.log(1.0/wh.vocab_size)*seq_length
    loss = 0
    for i in range(num_batches-1):
        s1 = i * seq_length
        s2 = (i+1) * seq_length
        c_input = corpus[s1:s2]
        c_output = corpus[s1+1:s2+1]
        
        batch_input = []
        batch_output = []
        for j in range(len(c_input)):
            c = c_input[j]
            c2 = c_output[j]
            batch_input.append(wh.char2id(c))
            batch_output.append(wh.id2onehot(wh.char2id(c2)))
            
        loss += back_prop(batch_input,batch_output,lr)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if not _ % decay_epoch:
        lr *= decay_rate
    if not _ % 10:
        print("Completed iteration:",_,"Cost: ",smooth_loss,"Learning Rate:",lr)
        predictTest()
 

## WORKING
##for _ in range(n_epochs):
##    smooth_loss = -np.log(1.0/wh.vocab_size)*seq_length
##    loss = 0
##    for i in range(corpus_len-1):
##        c = corpus[i]
##        c_next = corpus[i+1]
##        loss += back_prop(wh.char2id(c),wh.id2onehot(wh.char2id(c_next)),lr)
##    smooth_loss = smooth_loss * 0.999 + loss * 0.001
##    if not _ % decay_epoch:
##        lr *= decay_rate
##    if not _ % 1:
##        print("Completed iteration:",_,"Cost: ",smooth_loss,"Learning Rate:",lr)
##        predictTest()

       



 
        
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


